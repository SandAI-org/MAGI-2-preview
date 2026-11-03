"""Fake NVFP4 quantisation (quantise -> dequantise in bf16 compute) for scoring, no speed-up.

NVFP4 recipe as used by the Blackwell tensor cores: E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6}
times a per-16-element E4M3 block scale along the GEMM reduction dim, times one FP32 scale
per tensor (per expert matrix for weights, per activation tensor here). Rounding is nearest.
"""
import torch

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_BLOCK = 16
_FP8_MAX = 448.0


def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(a, _MID.to(a.device))
    return _E2M1.to(a.device)[idx]


def nvfp4_fake_quant(x: torch.Tensor, dim: int, global_amax: torch.Tensor | None = None,
                     rows_per_chunk: int = 1 << 18) -> torch.Tensor:
    """Return x quantised to NVFP4 along ``dim`` and dequantised back to x.dtype.
    ``global_amax``: per-tensor amax (scalar) or one per leading index (shape [x.shape[0]]) for
    per-expert weight scales; default = amax over the whole tensor."""
    xt = x.movedim(dim, -1)
    shape = xt.shape
    assert shape[-1] % _BLOCK == 0, shape
    flat = xt.reshape(-1, shape[-1])
    out = torch.empty_like(flat)
    if global_amax is None:
        global_amax = flat.abs().amax().float()
    g = global_amax.float().clamp_min(1e-12)
    if g.dim() == 1:  # per leading index
        per = flat.shape[0] // shape[0]
        g = g.repeat_interleave(per)
    else:
        g = g.expand(flat.shape[0])
    for r0 in range(0, flat.shape[0], rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, flat.shape[0])
        c = flat[r0:r1].float()
        gs = (g[r0:r1] / (6.0 * _FP8_MAX)).view(-1, 1, 1)  # fp32 per-tensor scale
        b = c.view(c.shape[0], -1, _BLOCK)
        amax = b.abs().amax(dim=-1, keepdim=True)
        bs = (amax / 6.0 / gs).clamp(max=_FP8_MAX).to(torch.float8_e4m3fn).float() * gs
        bs = torch.where(bs > 0, bs, torch.ones_like(bs))
        q = _round_e2m1((b / bs).abs()) * torch.sign(b) * bs
        out[r0:r1] = q.view(c.shape).to(out.dtype)
    return out.view(shape).movedim(-1, dim)
