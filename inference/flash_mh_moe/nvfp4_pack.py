"""NVFP4 packed expert weights (weight-only path): 4.5 bits/weight storage, Triton dequant to
bf16 per expert chunk in front of the bf16 grouped GEMM. Numerics are bit-identical to
``nvfp4_fake.nvfp4_fake_quant`` (same E2M1 rounding, E4M3 block scales, per-expert FP32 scale).

Layout is the tensor-core one so the same buffers can feed a real FP4 grouped GEMM later:
fp4 ``[G, N, K//2]`` uint8 (element 2i in the low nibble, along the GEMM reduction dim K),
sf ``[G, N, K//16]`` float8_e4m3fn, gs ``[G]`` fp32. The model stores experts as ``[G, K, N]``;
``dequant`` returns that layout as a transposed view of a contiguous ``[g, N, K]`` buffer.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .nvfp4_fake import _FP8_MAX, _MID

_BLOCK = 16


class FP4Experts:
    def __init__(self, fp4: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor):
        self.fp4, self.sf, self.gs = fp4, sf, gs
        G, N, KB = fp4.shape
        self.shape = (G, 2 * KB, N)  # model layout [G, K, N]
        self.device = fp4.device

    def size(self, d: int | None = None):
        return self.shape if d is None else self.shape[d]

    def dequant(self, g0: int, g1: int) -> torch.Tensor:
        """bf16 ``[g1-g0, K, N]`` (a transposed view of a contiguous ``[g, N, K]`` tensor)."""
        return dequant_nvfp4(self.fp4[g0:g1], self.sf[g0:g1], self.gs[g0:g1]).transpose(1, 2)

    def nbytes(self) -> int:
        return self.fp4.numel() + self.sf.numel() + self.gs.numel() * 4


def pack_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """w: ``[g, K, N]`` bf16/fp32 (CUDA). Per-expert global scale from the expert's amax."""
    g, K, N = w.shape
    assert K % _BLOCK == 0, K
    wt = w.transpose(1, 2).float().contiguous()  # [g, N, K]
    amax = wt.abs().amax(dim=(1, 2))
    gs = (amax / (6.0 * _FP8_MAX)).clamp_min(1e-12)  # same expression as nvfp4_fake
    b = wt.view(g, N, K // _BLOCK, _BLOCK)
    bamax = b.abs().amax(dim=-1, keepdim=True)
    sf = (bamax / 6.0 / gs.view(g, 1, 1, 1)).clamp(max=_FP8_MAX).to(torch.float8_e4m3fn)
    bs = sf.float() * gs.view(g, 1, 1, 1)
    bs = torch.where(bs > 0, bs, torch.ones_like(bs))
    idx = torch.bucketize((b / bs).abs(), _MID.to(w.device)).to(torch.uint8)
    code = (idx | ((b < 0).to(torch.uint8) << 3)).view(g, N, K)
    fp4 = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    return fp4, sf.view(g, N, K // _BLOCK).contiguous(), gs.contiguous()


@triton.jit
def _dequant_kernel(fp4_ptr, sf_ptr, gs_ptr, out_ptr, N, KB: tl.constexpr, K: tl.constexpr, SFK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per (expert, n) row of K elements
    g = row // N
    gs = tl.load(gs_ptr + g)
    offs = tl.arange(0, BLOCK)
    m = offs < KB
    b = tl.load(fp4_ptr + row * KB + offs, mask=m, other=0)
    sf = tl.load(sf_ptr + row * SFK + offs // 8, mask=m, other=0.0).to(tl.float32)
    scale = sf * gs
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    # E2M1 decode: bit3 sign, bits 2-1 exponent, bit0 mantissa
    e_lo = (lo >> 1) & 3
    e_hi = (hi >> 1) & 3
    m_lo = (lo & 1).to(tl.float32)
    m_hi = (hi & 1).to(tl.float32)
    mag_lo = tl.where(e_lo == 0, m_lo * 0.5, (1.0 + 0.5 * m_lo) * tl.exp2((e_lo - 1).to(tl.float32)))
    mag_hi = tl.where(e_hi == 0, m_hi * 0.5, (1.0 + 0.5 * m_hi) * tl.exp2((e_hi - 1).to(tl.float32)))
    v_lo = tl.where(((lo >> 3) & 1) == 1, -mag_lo, mag_lo) * scale
    v_hi = tl.where(((hi >> 3) & 1) == 1, -mag_hi, mag_hi) * scale
    tl.store(out_ptr + row * K + 2 * offs, v_lo.to(tl.bfloat16), mask=m)
    tl.store(out_ptr + row * K + 2 * offs + 1, v_hi.to(tl.bfloat16), mask=m)


def dequant_nvfp4(fp4: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """fp4 ``[g, N, K//2]``, sf ``[g, N, K//16]``, gs ``[g]`` -> bf16 ``[g, N, K]`` contiguous."""
    g, N, KB = fp4.shape
    K = 2 * KB
    out = torch.empty(g, N, K, dtype=torch.bfloat16, device=fp4.device)
    _dequant_kernel[(g * N,)](fp4, sf, gs, out, N, KB=KB, K=K, SFK=K // _BLOCK, BLOCK=triton.next_power_of_2(KB))
    return out
