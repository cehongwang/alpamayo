"""Thin wrapper around alpamayo_r1.eval that forces the VLM attention impl to
``sdpa`` and surfaces full tracebacks for TRT per-clip failures.

The checkpoint config requests ``flash_attention_2``, but the prebuilt
flash-attn binary in this venv is ABI-incompatible with the currently installed
torch nightly (undefined torch C++ symbol on import). ``sdpa`` is exact
attention and is also what the TRT vision/LM paths use, so it is the correct,
consistent reference for an eager-vs-TRT parity comparison.

Usage: same CLI args as eval.py, e.g.
    PYTHONPATH=src python run_eval_sdpa.py --ckpt ... --greedy --save_results ...
"""
import functools
import sys
import traceback

from alpamayo_r1.models import base_model

_orig_init = base_model.ReasoningVLAConfig.__init__


def _patched_init(self, *args, **kwargs):
    kwargs["attn_implementation"] = "sdpa"
    _orig_init(self, *args, **kwargs)


base_model.ReasoningVLAConfig.__init__ = _patched_init

import alpamayo_r1.eval as E

_orig_trt = E.compute_minade_for_clip_trt


@functools.wraps(_orig_trt)
def _trt_with_tb(*args, **kwargs):
    try:
        return _orig_trt(*args, **kwargs)
    except Exception:
        print("=" * 70, file=sys.stderr)
        print("FULL TRACEBACK for compute_minade_for_clip_trt:", file=sys.stderr)
        traceback.print_exc()
        print("=" * 70, file=sys.stderr)
        raise


E.compute_minade_for_clip_trt = _trt_with_tb

if __name__ == "__main__":
    import torch

    with torch.no_grad():
        E.main()
