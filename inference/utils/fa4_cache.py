# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Persist MagiAttention's FA4 (CuTe DSL) JIT kernels across runs.

MagiAttention loads pre-compiled kernels from ``MAGI_ATTENTION_FFA_FA4_CACHE_DIR`` at import
time but only its ``precompile_ffa_fa4`` tool writes them, for a fixed config sweep. The
compile key carries no sequence length, so the kernels a real run JIT-compiled are the
complete set: export them at the end of the run and the next one starts warm.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import subprocess


def export_ffa_fa4_cache(cache_dir: str | None = None) -> int:
    try:
        import cutlass.cute as cute
        from magi_attention.functional import fa4_utils as U
    except Exception as e:  # noqa: BLE001
        print(f"[magi2] FA4 cache export skipped: {e}", flush=True)
        return 0
    cache_dir = cache_dir or U.FFA_FA4_CACHE_DIR
    runtime_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=True)
    n = 0
    for name, meta in U.COMPILED_META_DICT.items():
        for key, kernel in meta["cache_dict"].items():
            if not hasattr(kernel, "export_to_c"):
                continue  # already a loaded pre-compiled module
            h = int(hashlib.sha256(f"{name}_{key}".encode("utf-8")).hexdigest(), 16)
            d = os.path.join(cache_dir, name, str(h))
            if os.path.exists(os.path.join(d, "kernel_lib.so")):
                continue
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "compiled_key.pkl"), "wb") as f:
                pickle.dump(key, f)
            func_name = f"kernel_{h}"
            with open(os.path.join(d, U._KERNEL_FUNC_NAME_FILE), "w") as f:
                f.write(func_name)
            obj = os.path.join(d, "kernel_obj.o")
            kernel.export_to_c(obj, function_name=func_name)
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o", os.path.join(d, "kernel_lib.so"), obj, *runtime_libs],
                check=True,
            )
            n += 1
    print(f"[magi2] FA4 cache: exported {n} new kernel(s) to {cache_dir}", flush=True)
    return n
