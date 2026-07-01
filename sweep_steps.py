#!/usr/bin/env python3
"""Sweep diffusion inference steps 5..10 over a fixed clip set, loading the model once."""
import json
import os
import sys

import numpy as np
import torch

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.eval import read_clip_ids_from_parquet, compute_minade_for_clip_pytorch

CKPT = "/home/trt-edgellm-alpamayo/tensorrt-edgellm-workspace/Alpamayo-R1-10B"
PARQUET = "/home/alpamayo/src/alpamayo_r1/1005_7cam_gold_eval_metadb_public.parquet"
T0_US = 5_100_000
LIMIT = 20
TOP_P = 0.98
TEMPERATURE = 0.6
NUM_TRAJ_SAMPLES = 6
MAX_GEN_LEN = 256
SEED = 42
DEVICE = "cuda"
STEPS = [int(x) for x in os.environ.get("AR1_SWEEP_STEPS", "5,6,7,8,9,10").split(",")]
OUT_PATH = os.environ.get("AR1_SWEEP_OUT", "/home/alpamayo/sweep_results.json")


def main():
    clip_ids = read_clip_ids_from_parquet(PARQUET)[:LIMIT]
    print(f"Loaded {len(clip_ids)} clip_ids", flush=True)

    model = AlpamayoR1.from_pretrained(CKPT, dtype=torch.float16).to(
        device=DEVICE, dtype=torch.float16
    )
    model.expert.config._attn_implementation = "sdpa"
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    results = {}
    per_clip_all = {}
    with torch.no_grad():
        for n in STEPS:
            os.environ["AR1_INFERENCE_STEP"] = str(n)
            per, ms = [], []
            for cid in clip_ids:
                try:
                    minade, t, _ = compute_minade_for_clip_pytorch(
                        model=model,
                        processor=processor,
                        clip_id=cid,
                        t0_us=T0_US,
                        top_p=TOP_P,
                        temperature=TEMPERATURE,
                        num_traj_samples=NUM_TRAJ_SAMPLES,
                        max_generation_length=MAX_GEN_LEN,
                        device=DEVICE,
                        seed=SEED,
                    )
                    per.append(minade)
                    ms.append(t)
                except Exception as e:
                    print(f"  FAILED step={n} clip={cid}: {e!r}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            avg_minade = float(np.mean(per)) if per else float("nan")
            avg_ms = float(np.mean(ms)) if ms else float("nan")
            results[n] = {
                "avg_minADE_m": avg_minade,
                "avg_ms_per_clip": avg_ms,
                "n_ok": len(per),
                "n_total": len(clip_ids),
            }
            per_clip_all[n] = per
            print(
                f"step={n}: avg_minADE={avg_minade:.4f} m  avg_time={avg_ms:.1f} ms/clip  "
                f"({len(per)}/{len(clip_ids)} ok)",
                flush=True,
            )

    out = {"config": {"limit": LIMIT, "seed": SEED, "ckpt": CKPT}, "results": results}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("RESULTS_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
