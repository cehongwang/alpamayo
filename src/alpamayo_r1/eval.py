#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import math
import time
from pathlib import Path

import pandas as pd

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.test_trt_torch import prepare_model_inputs
from alpamayo_r1.trt.compile_trt import compile_trt_modules, run_inference_trt





def resolve_vlm_name_or_path(ckpt_path: Path, override: str | None) -> str:
    """Prefer an explicit override, otherwise repair stale local checkpoint configs."""
    if override:
        return override

    config_path = ckpt_path / "config.json"
    if not config_path.is_file():
        return helper.BASE_PROCESSOR_NAME

    with config_path.open() as f:
        config = json.load(f)

    vlm_name_or_path = config.get("vlm_name_or_path")
    if not isinstance(vlm_name_or_path, str) or not vlm_name_or_path:
        return helper.BASE_PROCESSOR_NAME

    if Path(vlm_name_or_path).is_absolute() and not Path(vlm_name_or_path).exists():
        print(
            "Embedded vlm_name_or_path is missing; "
            f"falling back to {helper.BASE_PROCESSOR_NAME}: {vlm_name_or_path}"
        )
        return helper.BASE_PROCESSOR_NAME

    return vlm_name_or_path

def read_clip_ids_from_parquet(parquet_path: str) -> list[str]:
    """
    Reads clip_ids from parquet. Tries common column names; falls back to index if needed.
    Returns clip_ids as a list of strings (unique, preserving first occurrence order).
    """
    parquet_path = str(parquet_path)
    df = pd.read_parquet(parquet_path)
    cols_lower = {c.lower(): c for c in df.columns}
    clip_ids = df[cols_lower["key"]].astype(str).tolist()

    seen = set()
    uniq = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def _reasoning_from_extra(extra: dict) -> dict[str, str]:
    """Pull the (deterministic under greedy) reasoning text out of the extra dict.

    Values are arrays of shape [B, num_traj_sets, num_traj_samples]; we take the
    first sample since all samples share the same VLM rollout under greedy decode.
    """
    out: dict[str, str] = {}
    for key in ("cot", "meta_action", "answer"):
        if key not in extra:
            out[key] = ""
            continue
        val = np.asarray(extra[key]).reshape(-1)
        out[key] = str(val[0]) if val.size else ""
    return out


@torch.inference_mode()
def compute_minade_for_clip_pytorch(
    model: AlpamayoR1,
    processor,
    clip_id: str,
    t0_us: int,
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    device: str = "cuda",
    seed: int | None = 42,
    top_k: int | None = None,
) -> tuple[float, float, dict]:
    """
    Returns (minADE in meters, elapsed_ms, details) for one clip.

    ``details`` carries the reasoning text and the best-sample predicted XY
    trajectory so an eager run can later be compared against a TRT run.
    """
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)

    # Build chat message and tokenize
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    if seed is not None:
        # make sampling more stable/reproducible across clips
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # GT: (T,2)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()  # (T,2)

    # pred_xyz: assume (B, num_traj_sets, num_traj_samples, T, 3)
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (S,T,2)

    # ADE per sample: mean over time of L2 in XY
    d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)  # (S,T)
    ade = d.mean(axis=-1)  # (S,)
    best = int(ade.argmin())
    min_ade = float(ade[best])
    details = {
        "reasoning": _reasoning_from_extra(extra),
        "best_pred_xy": pred_xy[best].tolist(),  # (T,2)
    }
    return min_ade, elapsed_ms, details


@torch.inference_mode()
def compute_minade_for_clip_trt(
    model: AlpamayoR1,
    clip_id: str,
    t0_us: int,
    num_traj_samples: int,
    max_generation_length: int,
    trt_vision,
    trt_lm,
    trt_diffusion,
    device: str = "cuda",
    seed: int | None = 42,
) -> tuple[float, float, dict]:
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    create_inputs_fn = prepare_model_inputs(model, data, messages, device=device)
    seed = 42 if seed is None else seed
    pred_xyz, _, extra, elapsed_sec = run_inference_trt(
        model,
        create_inputs_fn,
        trt_vision=trt_vision,
        trt_lm=trt_lm,
        trt_diffusion=trt_diffusion,
        seed=seed,
        num_traj_samples=num_traj_samples,
        max_generation_length=max_generation_length,
    )

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()  # (T,2)
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (S,T,2)
    d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)  # (S,T)
    ade = d.mean(axis=-1)  # (S,)
    best = int(ade.argmin())
    details = {
        "reasoning": _reasoning_from_extra(extra),
        "best_pred_xy": pred_xy[best].tolist(),  # (T,2)
    }
    return float(ade[best]), elapsed_sec * 1000.0, details


def _token_prefix_agreement(a: str, b: str) -> float:
    """Fraction of leading whitespace-tokens that match between two strings."""
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    matched = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        matched += 1
    return matched / max(len(ta), len(tb))


def compare_results(eager_path: str, trt_path: str) -> dict:
    """Compute eager-vs-optimized parity metrics from two saved result files.

    Both files are expected to be produced by this script's ``--save_results``
    output (ideally with ``--greedy`` so the VLM rollout is deterministic).
    """
    from difflib import SequenceMatcher

    with open(eager_path) as f:
        eager = {r["clip_id"]: r for r in json.load(f)["results"]}
    with open(trt_path) as f:
        trt = {r["clip_id"]: r for r in json.load(f)["results"]}

    common = [c for c in eager if c in trt]
    if not common:
        raise ValueError("No overlapping clip_ids between the two result files.")

    # Per-field exact-match tallies, counted only over clips where at least one
    # side emitted the field. A field that neither run produces (e.g. this
    # checkpoint never emits meta_action/answer) would otherwise score a
    # misleading "" == "" == 1.0, so it is reported as N/A instead.
    text_match = {"cot": 0, "meta_action": 0, "answer": 0}
    text_eval_n = {"cot": 0, "meta_action": 0, "answer": 0}
    cot_prefix_sum = cot_sim_sum = 0.0
    cot_prefix_n = 0
    traj_l2_sum = minade_abs_sum = 0.0
    traj_count = 0

    for cid in common:
        re_, rt = eager[cid]["reasoning"], trt[cid]["reasoning"]
        for field in text_match:
            se, st = re_.get(field, ""), rt.get(field, "")
            if not se and not st:
                continue  # neither run produced this field; skip, don't score
            text_eval_n[field] += 1
            text_match[field] += int(se == st)

        ce, ct = re_.get("cot", ""), rt.get("cot", "")
        if ce or ct:
            cot_prefix_sum += _token_prefix_agreement(ce, ct)
            cot_sim_sum += SequenceMatcher(None, ce, ct).ratio()
            cot_prefix_n += 1

        minade_abs_sum += abs(eager[cid]["minade"] - trt[cid]["minade"])
        xe, xt = eager[cid].get("best_pred_xy"), trt[cid].get("best_pred_xy")
        if xe is not None and xt is not None:
            xe, xt = np.asarray(xe), np.asarray(xt)
            if xe.shape == xt.shape:
                traj_l2_sum += float(np.linalg.norm(xe - xt, axis=-1).mean())
                traj_count += 1

    n = len(common)

    def _rate(field: str) -> float | None:
        d = text_eval_n[field]
        return (text_match[field] / d) if d else None

    return {
        "num_compared": n,
        "meta_action_exact_match_rate": _rate("meta_action"),
        "meta_action_num_evaluated": text_eval_n["meta_action"],
        "answer_exact_match_rate": _rate("answer"),
        "answer_num_evaluated": text_eval_n["answer"],
        "cot_exact_match_rate": _rate("cot"),
        "cot_num_evaluated": text_eval_n["cot"],
        "cot_token_prefix_agreement": (cot_prefix_sum / cot_prefix_n) if cot_prefix_n else None,
        "cot_string_similarity": (cot_sim_sum / cot_prefix_n) if cot_prefix_n else None,
        "mean_abs_minade_delta_m": minade_abs_sum / n,
        "mean_best_traj_l2_m": (traj_l2_sum / traj_count) if traj_count else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default="1005_7cam_gold_eval_metadb_public.parquet")
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--ckpt", type=str, default="nvidia/Alpamayo-R1-10B")
    ap.add_argument(
        "--vlm_name_or_path",
        type=str,
        default=None,
        help="Optional override for the VLM processor/config path or Hub id.",
    )
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Deterministic argmax (top_k=1) VLM decode for reproducible reasoning text. "
        "Combine with --seed for eager-vs-TRT parity comparison.",
    )
    ap.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Path to write per-clip JSON (minADE + reasoning text + best trajectory).",
    )
    ap.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("EAGER_JSON", "TRT_JSON"),
        default=None,
        help="Compare two saved result files for eager-vs-optimized parity, then exit.",
    )
    ap.add_argument("--limit", type=int, default=644, help="How many unique clip_ids to evaluate.")
    ap.add_argument("--seed", type=int, default=42, help="Set -1 to disable reseeding per clip.")
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument(
        "--gc_every",
        type=int,
        default=1,
        help="Run Python garbage collection every N clips (0 disables).",
    )
    ap.add_argument(
        "--empty_cache_every",
        type=int,
        default=1,
        help="Call torch.cuda.empty_cache() every N clips (0 disables).",
    )
    ap.add_argument(
        "--compile_trt",
        action="store_true",
        help="Compile TRT vision/language/diffusion path before running evaluation.",
    )
    ap.add_argument(
        "--offload_module_to_cpu",
        action="store_true",
        help="Enable Torch-TensorRT offload_module_to_cpu for TRT module compile (LM/diffusion; default: disabled).",
    )
    ap.add_argument(
        "--trt_max_seq_len",
        type=int,
        default=0,
        help="Override max_seq_len for TRT LM compile (0 = use observed prefix + max_generation_length).",
    )
    ap.add_argument(
        "--trt_max_prefix_len",
        type=int,
        default=0,
        help="Override max_prefix_len for TRT compile (0 = use observed prefix_seq_len).",
    )
    args = ap.parse_args()

    if args.compare is not None:
        metrics = compare_results(args.compare[0], args.compare[1])
        print("============================================================")
        print("Eager-vs-optimized parity")
        print("============================================================")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    script_dir = Path(__file__).resolve().parent
    parquet_path = (script_dir / args.parquet).resolve()

    clip_ids = read_clip_ids_from_parquet(str(parquet_path))
    if args.limit is not None and args.limit > 0:
        clip_ids = clip_ids[: args.limit]

    print(f"Loaded {len(clip_ids)} clip_ids from: {parquet_path}")

    device = "cuda"
    ckpt_path = args.ckpt
    print(f"Loading checkpoint from: {ckpt_path}")
    model = AlpamayoR1.from_pretrained(
        str(ckpt_path),
        dtype=torch.float16,
    ).to(
        device=device, dtype=torch.float16
    )
    # The diffusion expert uses a custom 4D float additive attention mask that is
    # incompatible with flash-attention's unpadding path; force sdpa like the TRT paths.
    model.expert.config._attn_implementation = "sdpa"
    model.eval()
    seed = None if args.seed < 0 else args.seed

    trt_vision = None
    trt_lm = None
    trt_diffusion = None

    if args.compile_trt:
        # Use first clip as compile sample input for vision.
        compile_clip_id = clip_ids[0]
        compile_data = load_physical_aiavdataset(compile_clip_id, t0_us=args.t0_us)
        compile_messages = helper.create_message(compile_data["image_frames"].flatten(0, 1))
        compile_create_inputs_fn = prepare_model_inputs(
            model,
            compile_data,
            compile_messages,
            device=device,
        )

        seed_for_compile = 42 if seed is None else seed
        trt_lm_max_seq_len = args.trt_max_seq_len if args.trt_max_seq_len > 0 else None
        trt_max_prefix_len = args.trt_max_prefix_len if args.trt_max_prefix_len > 0 else None
        trt_vision, trt_lm, trt_diffusion, prefix_seq_len = compile_trt_modules(
            model=model,
            create_inputs_fn=compile_create_inputs_fn,
            seed=seed_for_compile,
            offload_module_to_cpu=args.offload_module_to_cpu,
            max_generation_length=args.max_generation_length,
            lm_max_seq_len=trt_lm_max_seq_len,
            max_prefix_len=trt_max_prefix_len,
            num_traj_samples=args.num_traj_samples,
        )
        # Release compile-only references early.
        del compile_data, compile_messages, compile_create_inputs_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # IMPORTANT: build processor once (do NOT rebuild per clip)
    processor = helper.get_processor(model.tokenizer)

    # Optional: tqdm progress if available
    try:
        from tqdm import tqdm
        it = tqdm(clip_ids, desc="Evaluating clips")
    except Exception:
        it = clip_ids

    per_clip = []
    per_clip_ms = []
    failed = []
    results = []

    # top_k=1 turns the existing multinomial decode into deterministic argmax.
    top_k = 1 if args.greedy else None
    if args.greedy:
        print("Greedy decode enabled (top_k=1): VLM reasoning text is deterministic per seed.")

    for i, clip_id in enumerate(it, start=1):

        # if i > 20: break
        try:
            if args.compile_trt:
                minade, elapsed_ms, details = compute_minade_for_clip_trt(
                    model=model,
                    clip_id=clip_id,
                    t0_us=args.t0_us,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length,
                    trt_vision=trt_vision,
                    trt_lm=trt_lm,
                    trt_diffusion=trt_diffusion,
                    device=device,
                    seed=seed,
                )
            else:
                minade, elapsed_ms, details = compute_minade_for_clip_pytorch(
                    model=model,
                    processor=processor,
                    clip_id=clip_id,
                    t0_us=args.t0_us,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length,
                    device=device,
                    seed=seed,
                    top_k=top_k,
                )
            per_clip.append(minade)
            per_clip_ms.append(elapsed_ms)
            if args.save_results:
                results.append(
                    {
                        "clip_id": clip_id,
                        "minade": minade,
                        "reasoning": details["reasoning"],
                        "best_pred_xy": details["best_pred_xy"],
                    }
                )

            if args.print_every and (i % args.print_every == 0):
                avg_so_far = float(np.mean(per_clip)) if per_clip else math.nan
                print(
                    f"[{i}/{len(clip_ids)}] clip_id={clip_id} "
                    f"minADE={minade:.4f}m time={elapsed_ms:.2f}ms | avg_so_far={avg_so_far:.4f}m"
                )

        except Exception as e:
            failed.append((clip_id, repr(e)))
            # try to recover GPU memory if something went wrong mid-inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.print_every:
                print(f"[{i}/{len(clip_ids)}] FAILED clip_id={clip_id}: {e}")
        finally:
            if args.gc_every > 0 and (i % args.gc_every == 0):
                gc.collect()
            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and (i % args.empty_cache_every == 0)
            ):
                torch.cuda.empty_cache()


    if per_clip:
        avg_minade = float(np.mean(per_clip))
        avg_time_ms = float(np.mean(per_clip_ms))
        print("============================================================")
        print(f"Average minADE over {len(per_clip)}/{len(clip_ids)} clips: {avg_minade:.6f} meters")
        print(f"Average eval time: {avg_time_ms:.2f} ms/clip")
    else:
        print("No successful clips; average minADE not computed.")

    if failed:
        print("============================================================")
        print(f"Failed clips: {len(failed)}")
        # print a few
        for cid, err in failed[:10]:
            print(f"  {cid}: {err}")
        if len(failed) > 10:
            print("  ...")

    if args.save_results:
        out_path = Path(args.save_results)
        payload = {
            "meta": {
                "path": "trt" if args.compile_trt else "eager",
                "greedy": bool(args.greedy),
                "seed": seed,
                "num_traj_samples": args.num_traj_samples,
                "max_generation_length": args.max_generation_length,
                "avg_minade": float(np.mean(per_clip)) if per_clip else None,
            },
            "results": results,
        }
        with out_path.open("w") as f:
            json.dump(payload, f)
        print(f"Saved {len(results)} per-clip results to: {out_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
