# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""MAGI-2 inference entry point.

Quick start:
    # Image-to-video, 1080p (prompt from file)
    torchrun --nproc_per_node=8 inference/pipeline/entry.py \
        --resolution 1080p --image assets/sample_000.png \
        --prompt-file assets/sample_000.txt --output output/

    # Text-to-video, 1080p (inline prompt)
    torchrun --nproc_per_node=8 inference/pipeline/entry.py \
        --resolution 1080p --prompt "A red fox in snow" --output output/

    # Batch from JSON (can mix i2v and t2v):
    torchrun --nproc_per_node=8 inference/pipeline/entry.py \
        --resolution 1080p --samples assets/demo_samples.json --output output/
"""

import argparse
import os

import torch
import torch.distributed as dist

from inference.common.magi2_config import Magi2Config, load_config
from inference.infra.distributed import initialize_expert_parallel, initialize_model_parallel
from inference.utils import print_rank_0

RESOLUTION_PRESETS = {
    "272p": {"config": "configs/magi2_preview.json", "pw": 448, "ph": 256},
    "540p": {"config": "configs/magi2_preview.json", "pw": 896, "ph": 512},
    "1080p": {"config": "configs/magi2_refiner.json", "pw": 896, "ph": 512, "rw": 1920, "rh": 1088},
}


def parse_args():
    p = argparse.ArgumentParser(description="MAGI-2 video generation")
    p.add_argument("--resolution", choices=list(RESOLUTION_PRESETS), default="1080p")
    p.add_argument("--prompt", type=str, nargs="*", default=None,
                   help="Inline prompt text(s). Mutually exclusive with --prompt-file.")
    p.add_argument("--prompt-file", type=str, nargs="*", default=None,
                   help="Path(s) to .txt file(s) containing prompts.")
    p.add_argument("--image", type=str, nargs="*", default=None,
                   help="First-frame image(s) for i2v. Omit for text-to-video.")
    p.add_argument("--samples", type=str, default=None,
                   help="Path to JSON file describing a batch of samples. "
                   "Each entry: {prompt|prompt_file, image?}. Mutually exclusive with --prompt/--image.")
    p.add_argument("--output", type=str, default="output", help="Output directory.")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="Clip duration. 10s is the only duration the model supports.")
    p.add_argument("--seed", type=int, default=42)
    # Advanced overrides
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--preview-width", type=int, default=None)
    p.add_argument("--preview-height", type=int, default=None)
    p.add_argument("--refiner-width", type=int, default=None)
    p.add_argument("--refiner-height", type=int, default=None)
    p.add_argument("--output-width", type=int, default=None)
    p.add_argument("--output-height", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--refiner-num-inference-steps", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (also triggered by MAGI2_DETERMINISTIC=1)")
    return p.parse_args()


def _resolve_prompts(args) -> list[dict]:
    """Build list of {prompt, image_path} from CLI args."""
    prompts: list[str] = []
    if args.prompt_file:
        for pf in args.prompt_file:
            prompts.append(open(pf).read().strip())
    elif args.prompt:
        prompts = list(args.prompt)
    else:
        prompts = ["A person speaks naturally to the camera."]

    images = args.image or [None] * len(prompts)
    if len(images) == 1 and len(prompts) > 1:
        images = images * len(prompts)
    assert len(images) == len(prompts), (
        f"Number of images ({len(images)}) must match prompts ({len(prompts)})"
    )
    return [{"prompt": p, "image_path": img} for p, img in zip(prompts, images)]



def _load_samples(path: str) -> list[dict]:
    """Load batch samples from a JSON file.

    Each entry is ``{"prompt": "...", "image": "path"}`` or
    ``{"prompt_file": "path.txt", "image": "path"}``.
    Omit ``image`` for text-to-video.
    """
    import json
    with open(path) as f:
        entries = json.load(f)
    samples = []
    for entry in entries:
        if "prompt_file" in entry:
            prompt = open(entry["prompt_file"]).read().strip()
        else:
            prompt = entry["prompt"]
        samples.append({"prompt": prompt, "image_path": entry.get("image")})
    return samples


def _init_distributed(config: Magi2Config):
    if not dist.is_initialized():
        dist.init_process_group(backend=config.engine_config.distributed_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(
        cp_size=config.engine_config.cp_size,
        distributed_timeout_minutes=config.engine_config.distributed_timeout_minutes,
    )
    if config.engine_config.ep_size > 1:
        initialize_expert_parallel(config.engine_config.ep_size)
    print_rank_0(
        f"[magi2] distributed: world={dist.get_world_size()} "
        f"cp={config.engine_config.cp_size} ep={config.engine_config.ep_size}"
    )


def _enable_deterministic(seed: int):
    import random, numpy as np
    os.environ["MAGI_ATTENTION_DETERMINISTIC_MODE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Implies cudnn.deterministic=True; no separate cudnn settings needed.
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.use_deterministic_algorithms(True)
    # PyTorch 2.9: use_deterministic_algorithms does NOT propagate to Inductor
    # compile-time autotuning. Manually disable configs that cause run-to-run
    # non-determinism under parallel GPU load.
    import torch._inductor.config as _ic
    _ic.dynamic_scale_rblock = False
    _ic.shape_padding = False
    _ic.comprehensive_padding = False
    _ic.benchmark_epilogue_fusion = False
    _ic.coordinate_descent_tuning = False
    # Deterministic env for vendored Triton MoE kernel is MAGI2_DETERMINISTIC (set above)
    print_rank_0(f"[magi2] deterministic mode enabled, seed={seed}")


def main():
    args = parse_args()
    preset = RESOLUTION_PRESETS[args.resolution]

    config = load_config(args.config or preset["config"])
    config.engine_config.seed = args.seed
    if os.environ.get("MAGI2_DETERMINISTIC", "0") == "1" or getattr(args, "deterministic", False):
        _enable_deterministic(config.engine_config.seed)

    _init_distributed(config)

    from inference.pipeline.pipeline import Magi2Pipeline
    pipeline = Magi2Pipeline(config)

    pw = args.preview_width or preset["pw"]
    ph = args.preview_height or preset["ph"]
    rw = args.refiner_width if args.refiner_width is not None else preset.get("rw")
    rh = args.refiner_height if args.refiner_height is not None else preset.get("rh")

    samples = _load_samples(args.samples) if args.samples else _resolve_prompts(args)
    os.makedirs(args.output, exist_ok=True)

    for idx, sample in enumerate(samples):
        print_rank_0(f"[magi2] sample {idx+1}/{len(samples)}")
        seed = args.seed + idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pipeline.run_offline(
            prompt=sample["prompt"],
            image_path=sample["image_path"],
            save_path_prefix=f"{args.output}/sample_{idx:03d}",
            seconds=args.seconds,
            preview_width=pw,
            preview_height=ph,
            refiner_width=rw,
            refiner_height=rh,
            output_width=args.output_width,
            output_height=args.output_height,
            num_inference_steps=args.num_inference_steps,
            magi2_refiner_num_inference_steps=args.refiner_num_inference_steps,
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
