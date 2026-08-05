# MAGI-2

MAGI-2 Preview is a research preview of a unified audio-video generation model.
A single Transformer processes text, video and audio as one token sequence, and
an ultra-fine-grained mixture of experts holds around 114B parameters while
activating around 6B of them per token. The architecture, the training system
built around it, and the data pipeline are described in [MAGI-2 Preview: Scaling
Video Generation Models Efficiently](https://sand.ai/blog/magi-2-preview).

This repository is the inference code. It generates video from a text prompt
(T2V) or from a prompt plus a still image (I2V), with sound generated alongside
the video and muxed into the output file. Clips are 10 seconds long, which is
the only duration the model currently supports. Generation runs in two stages:
`magi2_preview` denoises the clip at low resolution, and `magi2_refiner` takes
that result up to 1080p. The refiner is optional; the preview stage alone
produces 272p or 540p video.

## Requirements

- NVIDIA Hopper GPUs. The 1080p preset expects 8 of them.
- Python 3.12 and a recent CUDA toolkit.
- `ffmpeg` on `PATH`, to mux the audio track. Without it the video is still
  written, just silently.

## Setup

### Docker

The published image already has the dependencies built, including the ones that
need a compiler:

```bash
docker pull sandai/magi2:latest
docker run --gpus all -it -v /path/to/ckpt:/workspace/ckpt sandai/magi2:latest
```

There is a tag per commit as well, `sandai/magi2:<commit>`. Name that one when
reporting a result, because `latest` moves; the image also records what it was
built from in `/etc/magi2-build-info`.

Building it yourself is only necessary to change a dependency version, or to
work somewhere the registry is not reachable:

```bash
docker build -t magi2:local .
```

### From source

```bash
pip install -r requirements.txt
```

MAGI-2 also needs [MagiAttention](https://github.com/SandAI-org/MagiAttention)
and MagiCompiler. The pinned revisions are recorded as build args in the
Dockerfile.

## Checkpoints

Weights are not bundled with the code. Put them under `ckpt/` in the repository
root, which is gitignored:

```
ckpt/
├── magi2_preview/                      # preview stage: safetensors shards + index
├── magi2_refiner/                      # refiner stage
├── Wan2.2-TI2V-5B/                     # video VAE
├── Qwen3.5-27B/                        # text encoder
└── turbo_vae/                          # fast VAE decoder
    ├── TurboV3-Wan22-TinyShallow_7_7.json
    └── checkpoint-340000.ckpt
```

| Directory | Contents |
| --- | --- |
| `magi2_preview` | Preview-stage transformer, released with MAGI-2 |
| `magi2_refiner` | Refiner-stage transformer, released with MAGI-2 |
| `Wan2.2-TI2V-5B` | Video VAE, from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| `Qwen3.5-27B` | Text encoder |
| `turbo_vae` | Distilled VAE decoder, used for decoding by default |

The configs under `configs/` reference these as `${MAGI2_CKPT_ROOT}/<name>`, and
that variable defaults to `<repo>/ckpt`. To keep weights somewhere else, point it
at them rather than editing the configs:

```bash
export MAGI2_CKPT_ROOT=/data/magi2-weights
```

## Prompts

The captions the model was trained on are long and structured, so a prompt
written by hand underuses it. Two system prompts for a prompt-enhancement LLM
are included: `prompts/t2v.md` for text to video, and `prompts/i2v.md` for a
prompt plus a still image.

Use one as the system prompt of an instruction-following model, pass the raw
prompt as the message (the still as well, for I2V), and feed the JSON caption it
returns to the pipeline in place of the prompt. Both lay out the 10 seconds the
model generates.

`assets/` has both ends of that step. `sample_000.txt` through `sample_002.txt`
are raw prompts, the kind you would hand to the enhancer;
`sample_enhanced_t2v.json` is the shape one comes back in. The demo batch runs
both, so enhancing is not a precondition for generating.

## Running inference

`scripts/run_demo.sh` picks a config and resolution preset, then launches
`inference/pipeline/entry.py` under `torchrun` on every visible GPU. It defaults
to 1080p, seed 42, and the batch in `assets/demo_samples.json`:

```bash
bash scripts/run_demo.sh                           # 1080p
RESOLUTION=540p bash scripts/run_demo.sh           # preview only, no refiner
SAMPLES=my_samples.json bash scripts/run_demo.sh   # a different batch
OUTPUT_DIR=output/run7 bash scripts/run_demo.sh
```

The script reads `RESOLUTION`, `SAMPLES`, `OUTPUT_DIR`, `SEED` and `MASTER_PORT`,
and nothing else. Videos land in `$OUTPUT_DIR/sample_000.mp4` and up, numbered by
position in the batch.

A samples file is a JSON array with one entry per video. An entry carries its
prompt inline as `prompt` or as a path in `prompt_file`, and a first frame in
`image`; leaving `image` out makes it a T2V entry. The shipped batch runs the
three stills in `assets/` as I2V, the same three prompts again as T2V, and
`assets/sample_enhanced_t2v.json`.

For a single clip, call the entry point directly:

```bash
torchrun --nproc_per_node=8 inference/pipeline/entry.py \
    --resolution 540p --prompt "a red fox in snow" --output output/
```

It also takes `--prompt-file`, `--image`, `--seed`, `--config`, the
`--preview-width` / `--preview-height` and `--refiner-width` / `--refiner-height`
pairs, `--output-width` / `--output-height`, `--num-inference-steps`,
`--refiner-num-inference-steps` and `--deterministic`. Of these only
`--resolution`, `--seed`, `--samples` and `--output` are reachable through
`run_demo.sh`.

The resolution presets are:

| `RESOLUTION` | Config | Preview | Refiner |
| --- | --- | --- | --- |
| `272p` | `configs/magi2_preview.json` | 256x448 | off |
| `540p` | `configs/magi2_preview.json` | 512x896 | off |
| `1080p` | `configs/magi2_refiner.json` | 512x896 | 1088x1920 |

`magi2_refiner.json` extends `magi2_preview.json` and carries only what the
refiner stage adds, so a shared setting is edited in one place.

A preset name is a delivery tier, not the shape that gets generated. The VAE
stride constrains every generated dimension to a multiple of 16, so the shape
lands near the tier rather than on it: the `272p` tier generates 448 tall and
`1080p` generates 1088 wide. Videos are written at that generated shape. Pass
`--output-width` and `--output-height` to have the finished video rescaled to an
exact size, the way the reference delivers its tiers: 270x480, 540x960 or
1080x1920.

Four environment variables decide where each large component sits between
phases: `MAGI2_TEXT_ENC_OFFLOAD_MODE`, `MAGI2_PREVIEW_OFFLOAD_MODE`,
`MAGI2_REFINER_OFFLOAD_MODE` and `MAGI2_VAE_OFFLOAD_MODE`, each one of `cpu`,
`gpu` or `roundtrip`. The preview and the refiner default to `roundtrip`, staged
in and out around the stage that needs them, because at 1080p neither fits on an
80GB card next to the other's activations.

Decoding uses the distilled turbo decoder from `ckpt/turbo_vae`, a temporal
sliding window that runs on one rank per video. `MAGI2_DETERMINISTIC=1`, or
`--deterministic`, makes the MoE scatter and the attention kernels bit-exact at
some cost in speed. `MAGI2_SAVE_LATENT_PATH` writes the post-refiner latent of
each sample to that directory.

## License

Apache 2.0. See [LICENSE](LICENSE).
