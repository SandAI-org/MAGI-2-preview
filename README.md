# MAGI-2

MAGI-2 generates video from a text prompt (T2V) or from a prompt plus a still
image (I2V). Generation runs in two stages: `magi2_preview` denoises the clip at
low resolution, and `magi2_refiner` takes that result up to 1080p. The refiner is
optional; the preview stage alone produces 270p or 540p video.

## Requirements

- NVIDIA Hopper GPUs. The 1080p preset expects 8 of them.
- Python 3.12 and a recent CUDA toolkit.

## Setup

### Docker

The published image contains all compiled dependencies (flash-attention,
MagiAttention, MagiCompiler, cuDNN fix, etc.) but **not** the model code.
Clone the repo first, then mount it into the container:

```bash
# 1. Pull the image
docker pull sandai/magi-2-preview:latest

# 2. Clone the code
git clone https://github.com/SandAI-org/MAGI-2.git
cd MAGI-2

# 3. Run — mount code + checkpoints into /workspace
docker run --gpus all -it --ipc=host \
  -v $(pwd):/workspace \
  -v /path/to/ckpt:/workspace/ckpt \
  sandai/magi-2-preview:latest
```

There is a tag per commit as well, `sandai/magi-2-preview:<short-sha>`.
The image records what it was built from in `/etc/magi2-build-info`.

Building the image yourself is only necessary to change a dependency version
or to work somewhere the registry is not reachable:

```bash
docker build -t magi-2-preview:local .
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
├── magi2_preview/                      # preview stage transformer
├── magi2_refiner/                      # refiner stage transformer
├── Wan2.2-TI2V-5B/                     # video VAE
├── Qwen3.5-27B/                        # text encoder
├── turbo_vae/                          # distilled fast VAE decoder
└── stable-audio-open-1.0/             # audio VAE (Stable Audio Open)
```

| Directory | Contents |
| --- | --- |
| `magi2_preview` | Preview-stage transformer, released with MAGI-2 |
| `magi2_refiner` | Refiner-stage transformer, released with MAGI-2 |
| `Wan2.2-TI2V-5B` | Video VAE, from [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| `Qwen3.5-27B` | Text encoder |
| `turbo_vae` | Distilled VAE decoder, used for decoding by default |
| `stable-audio-open-1.0` | Audio VAE for audio decode/encode, from [Stability AI](https://huggingface.co/stabilityai/stable-audio-open-1.0) |

The configs under `configs/` reference these as `${MAGI2_CKPT_ROOT}/<name>`, and
that variable defaults to `<repo>/ckpt`. To keep weights somewhere else, point it
at them rather than editing the configs:

```bash
export MAGI2_CKPT_ROOT=/data/magi2-weights
```

## Running inference

`scripts/run_demo.sh` picks a config and resolution preset, then launches
`inference/pipeline/entry.py` under `torchrun`. It defaults to I2V at 1080p, 12
seconds, seed 42, on 8 GPUs:

```bash
bash scripts/run_demo.sh                              # I2V, 1080p
TASK=t2v bash scripts/run_demo.sh                     # text to video
RESOLUTION=540p bash scripts/run_demo.sh              # preview only, no refiner
PROMPT="a red fox in snow" bash scripts/run_demo.sh   # one ad-hoc prompt
```

Without `PROMPT`, prompts come from `assets/demo_samples.json`, three prompts
paired with the stills in `assets/`. `TASK=t2v` runs the same three
prompts and ignores the stills.
Videos and a `run.log` land in `output/<task>_<resolution>_<timestamp>/`.

The resolution presets are:

| `RESOLUTION` | Output (WxH) | Preview (WxH) | Refiner (WxH) |
| --- | --- | --- | --- |
| `272p` | 256x448 | 256x448 | off |
| `540p` | 512x896 | 512x896 | off |
| `1080p` | 1088x1920 | 512x896 | 1088x1920 |

The VAE stride constrains generated dimensions to multiples of 16, so the
output is 1088 wide rather than exactly 1080. To get a pixel-exact final
size (e.g. 1920x1080), pass `--output-width 1920 --output-height 1080`.

Everything else is overridable through the environment: `SECONDS_PER_VIDEO`,
`SEED`, `GPUS_PER_NODE`, `NUM_INFERENCE_STEPS`, `OUTPUT_DIR`, and the explicit
`PREVIEW_WIDTH` / `REFINER_WIDTH` pairs. Compilation caches are keyed on shapes and deterministic mode.

Three more decide where each large component sits: `MAGI2_TEXT_ENC_OFFLOAD_MODE`,
`MAGI2_PREVIEW_OFFLOAD_MODE` and `MAGI2_REFINER_OFFLOAD_MODE`, each one of `cpu`,
`gpu` or `roundtrip`. The defaults stage the preview and the refiner in and out
around the stage that needs them, because at 1080p neither fits on an 80GB card
next to the other's activations.

`MAGI2_VAE_DECODE_MODE` picks the decoder. Unset, or `none`, uses the distilled
turbo decoder from `ckpt/turbo_vae`: a temporal sliding window that runs on a
single rank. `tiled` and `chunk` use the full Wan VAE instead and spread the work
over the ranks sharing the video, `tiled` by splitting the frame into overlapping
spatial tiles and `chunk` by splitting each convolution along the width. They cost
more time and memory than the distilled decoder.

## License

Apache 2.0. See [LICENSE](LICENSE).
