# Running MAGI-2 Preview on Blackwell (sm_100 / sm_103)

The shipped image targets Hopper: FlashAttention-3, MagiAttention's FFA kernel and the fused
Triton MH-MoE kernel are all sm_90a-only. This branch makes the preview + refiner pipeline run
on B200/B300 (tested on an HGX 8x B300, sm_103, NGC PyTorch 25.10 / CUDA 13 / torch 2.9).

## Image

`docker/Dockerfile.blackwell` builds on `sandai/magi-2-preview:latest` and adds the
MagiAttention FA4 (CuTe DSL) backend from the `magi_attn_blackwell_support` branch of
[demonatic/flash-attention](https://github.com/demonatic/flash-attention), compiled for
sm_100/sm_103, plus the `openai` client that `requirements.txt` lists but the image lacks.

```
docker build -f docker/Dockerfile.blackwell -t magi-2-preview:blackwell .
```

## What changes on Blackwell

| Component | Hopper | Blackwell (this branch) |
|---|---|---|
| Preview attention (with sinks) | FA3 | MagiAttention FA4 (`MAGI2_PREVIEW_ATTN=fa4`, default when the backend is present; `fa2` falls back to FlashAttention-2 with the same LSE sink correction) |
| MH-MoE (36 layers, 12 heads x 256 experts) | fused Triton kernel | `torch._grouped_mm` backend (`MAGI2_MOE_BACKEND=grouped_mm`, automatic on cc >= 10): one Triton routing kernel (sigmoid + bias + top-k + normalise), flat gather, Triton SwiGLU7, gate/up/act/down per group chunk, combine through the inverse routing permutation (fp32, deterministic). cuBLAS caps a grouped GEMM at 1024 groups; calls are sliced at `MAGI2_GMM_MAX_GROUPS` (768) |
| Refiner windowed attention | MagiAttention FFA | FlashAttention-2 varlen over the range list (`inference/model/blocksparse_fa2.py`, `MAGI2_REFINER_ATTN=fa2`). FA4 (`=fa4`) works but materialises a dense (ranges x seqlen) table, 1.5 TB at 1080p |
| FA4 kernel cache | n/a | the CuTe kernels JIT at ~6 s per variant and MagiAttention only loads precompiled ones from `MAGI_ATTENTION_FFA_FA4_CACHE_DIR`; `inference/utils/fa4_cache.py` exports what a run compiled so the next run starts warm (`MAGI2_EXPORT_FA4_CACHE=0` disables) |

Triton 3.5 (the NGC 25.10 version) cannot lower the fused MoE kernel or FlexAttention for
sm_103 (MLIR dominance / TMEM / `tcgen05.wait.st` failures); the grouped-GEMM path does not
depend on Triton for the GEMMs.

## Configs and layouts

`configs/magi2_{preview,refiner}_{2,4}gpu.json` set `cp_size = ep_size = 2 | 4`. Expert
parallelism must divide 256, so 3 GPUs is not a layout. 1080p runs on 2 B300 with nothing
offloaded (peak reserved ~245 GB per rank).

## Performance (B300, 1080p clip = preview 100 steps + refiner 5 steps + VAE + audio)

| | 4 GPUs (cp=ep=4) | 2 GPUs (cp=ep=2) |
|---|---|---|
| preview denoise, 1080p | 5.2 min (3.1 s/it) | |
| preview denoise, 540p | | 5.0 min (3.0 s/it) |
| whole clip, steady state | ~12.5 min | ~7.1 min per clip at 540p |

A preview step on 2 GPUs at the 540p shape is ~48 % attention (FA4 near its dense roofline),
~34 % MoE (grouped GEMMs ~10 ms of ~18 ms per layer call), ~8 % dense projections.

## Numerics

Same-seed clips across kernel paths (FA2 vs FA4, Triton vs grouped-GEMM MoE) land at 24-27 dB
PSNR against each other: same composition with small drift, because the MoE router picks
top-6 of 256 experts by sigmoid scores whose 6th/7th gap is below 0.01 for most tokens, and any
rounding difference flips some of those picks. grouped-GEMM MoE vs an fp32 per-expert reference:
relative error 3.6e-3; range-sparse FA2 vs the repo's LSE-correction reference: max abs diff
2e-3 in bf16. `MAGI2_DETERMINISTIC=1` keeps the sequential scatter for bit-exact reruns on
the same kernel path.

## Debug knobs added

- `MAGI2_PROFILE_STEPS=a-b`: `torch.profiler` over preview steps a..b (rank 0 prints the
  kernel table and writes a chrome trace under `MAGI_COMPILE_CACHE_ROOT_DIR`).
- `MAGI2_PREVIEW_LATENT_CACHE=<file>`: save the preview latents after the first run and skip
  the preview denoise on later runs (refiner iteration).
- `MAGI2_DUMP_RANGES=<dir>`: dump the refiner's attention range lists.
- `MAGI2_MOE_SCATTER=inv|fp32|bf16`: combine variant (inverse-permutation default).
