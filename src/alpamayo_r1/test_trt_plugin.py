# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run PyTorch FP16 vs TRT-Plugin FP16 inference for the Alpamayo-R1 VLA.

Mirrors ``alpamayo1_5.test_trt_torch`` but drives the TRT pipeline through
the TRT-LLM ``xqa_attn`` plugin (``compile_trt_with_plugin`` +
``run_inference_trt_plugin``) instead of the generic SDPA wrapper.

Usage::

    python -m alpamayo_r1.test_trt_plugin
    python -m alpamayo_r1.test_trt_plugin --num_traj_samples 6
    python -m alpamayo_r1.test_trt_plugin --skip-trt      # baseline only
    python -m alpamayo_r1.test_trt_plugin --benchmark     # timing runs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_CONVERTER_DIR = os.path.realpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "TensorRT", "tools", "llm")
)
if os.path.isdir(_PLUGIN_CONVERTER_DIR) and _PLUGIN_CONVERTER_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_CONVERTER_DIR)

torch._dynamo.config.capture_scalar_outputs = True

from alpamayo_r1.trt.compile_trt import (
    compile_trt_with_plugin,
    run_inference_trt_plugin,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(clip_id: str, t0_us: int = 5_100_000) -> tuple[dict, list]:
    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    return data, messages


def prepare_model_inputs(
    model, data: dict, messages: list, device: str = "cuda"
) -> Callable[[], dict[str, Any]]:
    from alpamayo_r1 import helper

    processor = helper.get_processor(model.tokenizer)

    def create_inputs():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"].clone(),
            "ego_history_rot": data["ego_history_rot"].clone(),
        }
        return helper.to_device(model_inputs, device)

    return create_inputs


def compute_trajectory_metrics(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> dict[str, float]:
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    ade_per_sample = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    fde_per_sample = np.linalg.norm(pred_xy[:, :, -1] - gt_xy[:, -1], axis=1)
    return {
        "min_ade": float(ade_per_sample.min()),
        "mean_ade": float(ade_per_sample.mean()),
        "min_fde": float(fde_per_sample.min()),
        "mean_fde": float(fde_per_sample.mean()),
    }


def run_inference_pytorch(
    model,
    create_inputs_fn: callable,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()
    start_time = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    return pred_xyz, pred_rot, extra, time.perf_counter() - start_time


def benchmark_inference(
    run_fn: callable,
    num_runs: int = 5,
    warmup_runs: int = 2,
    label: str = "Model",
) -> float:
    logger.info("Benchmarking %s...", label)
    for _ in range(warmup_runs):
        run_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        _, _, _, elapsed = run_fn()
        torch.cuda.synchronize()
        times.append(elapsed)
    return sum(times) / len(times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRT-Plugin test for Alpamayo-R1")
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument("--clip_id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--num_traj_samples", type=int, default=1)
    parser.add_argument("--max_generation_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TRT compilation")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch baseline")
    parser.add_argument("--num-iterations", type=int, default=12, help="Total iterations (incl. warmup)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations to exclude from stats")
    return parser.parse_args()


def _print_timing(name: str, times_ms: list[float]) -> None:
    arr = np.array(times_ms)
    if len(arr) == 0:
        return
    print(
        f"  {name:<22} min={arr.min():7.1f}  avg={arr.mean():7.1f}  "
        f"max={arr.max():7.1f}  std={arr.std():6.1f}  (ms)"
    )


def _print_minade(name: str, ades: list[float]) -> None:
    arr = np.array(ades)
    if len(arr) == 0:
        return
    print(
        f"  {name:<22} min={arr.min():7.4f}  avg={arr.mean():7.4f}  "
        f"max={arr.max():7.4f}  std={arr.std():6.4f}  (m)"
    )


def main() -> int:
    args = parse_args()
    data, messages = load_test_data(args.clip_id)
    gt_xyz = data["ego_future_xyz"]

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    print(
        f"clip_id={args.clip_id}  num_traj_samples={args.num_traj_samples}  "
        f"iters={args.num_iterations}  warmup={args.warmup}  "
        f"skip_pytorch={args.skip_pytorch}  skip_trt={args.skip_trt}"
    )

    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.float16).to("cuda")
    model.eval()
    model.config.attn_implementation = "sdpa"

    create_inputs_fn = prepare_model_inputs(model, data, messages, device="cuda")

    # ── Compile TRT engines once (iter -1) ────────────────────────────
    trt_vision = trt_lm = trt_diffusion = plugin_info = None
    if not args.skip_trt:
        trt_vision, trt_lm, trt_diffusion, plugin_info = compile_trt_with_plugin(
            model,
            create_inputs_fn,
            seed=args.seed,
            max_generation_length=args.max_generation_length,
            num_traj_samples=args.num_traj_samples,
        )

    pt_times: list[float] = []
    pt_ades: list[float] = []
    trt_times: list[float] = []
    trt_ades: list[float] = []
    pt_coc = trt_coc = "(skipped)"

    for i in range(args.num_iterations):
        print(f"\n=== iter {i} (clip={args.clip_id}) ===", flush=True)

        # ── PyTorch FP16 baseline ────────────────────────────────────
        if not args.skip_pytorch:
            torch.cuda.synchronize(); t = time.perf_counter()
            pred_xyz_pt, _, extra_pt, _ = run_inference_pytorch(
                model,
                create_inputs_fn,
                seed=args.seed,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
            torch.cuda.synchronize(); pt_elapsed = 1000 * (time.perf_counter() - t)
            pt_metrics = compute_trajectory_metrics(pred_xyz_pt, gt_xyz)
            pt_coc = str(extra_pt["cot"][0][0, 0])
            pt_times.append(pt_elapsed)
            pt_ades.append(pt_metrics["min_ade"])
            print(f"  PyTorch    : {pt_elapsed:7.1f} ms   minADE={pt_metrics['min_ade']:.4f} m")

        # ── TRT Plugin FP16 ──────────────────────────────────────────
        if not args.skip_trt:
            torch.cuda.synchronize(); t = time.perf_counter()
            pred_xyz_trt, _, extra_trt, _ = run_inference_trt_plugin(
                model,
                create_inputs_fn,
                trt_vision=trt_vision,
                trt_lm=trt_lm,
                trt_diffusion=trt_diffusion,
                plugin_info=plugin_info,
                seed=args.seed,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
            torch.cuda.synchronize(); trt_elapsed = 1000 * (time.perf_counter() - t)
            trt_metrics = compute_trajectory_metrics(pred_xyz_trt, gt_xyz)
            trt_coc = str(extra_trt["cot"][0][0, 0])
            trt_times.append(trt_elapsed)
            trt_ades.append(trt_metrics["min_ade"])
            print(f"  TRT Plugin : {trt_elapsed:7.1f} ms   minADE={trt_metrics['min_ade']:.4f} m")

    print("\n" + "=" * 78)
    print(f"Summary  (warmup={args.warmup} / {args.num_iterations}, hot iters {args.warmup}–{args.num_iterations - 1})")
    print("=" * 78)
    if pt_times:
        _print_timing("PyTorch FP16",   pt_times[args.warmup :])
        _print_minade("PyTorch FP16",   pt_ades[args.warmup :])
    if trt_times:
        _print_timing("TRT Plugin FP16", trt_times[args.warmup :])
        _print_minade("TRT Plugin FP16", trt_ades[args.warmup :])

    if pt_times and trt_times:
        pt_avg = float(np.mean(pt_times[args.warmup :]))
        trt_avg = float(np.mean(trt_times[args.warmup :]))
        speedup = pt_avg / trt_avg if trt_avg > 0 else float("nan")
        print(f"\n  Speedup (TRT vs PyTorch): {speedup:5.2f}x   ({pt_avg:.1f} → {trt_avg:.1f} ms)")

    print("\nCoC outputs (last iter):")
    print(f"  PyTorch    : {pt_coc[:100]}...")
    print(f"  TRT Plugin : {trt_coc[:100]}...")

    return 0


if __name__ == "__main__":
    with torch.no_grad():
        raise SystemExit(main())
