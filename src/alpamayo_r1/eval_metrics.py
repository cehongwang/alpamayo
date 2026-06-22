#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy / fidelity metrics for the Alpamayo-R1 rollout.

Motivation
----------
``eval.py`` currently scores a clip with a single number: ``minADE`` (the
best-of-S, mean-over-horizon L2 displacement against the ground-truth future).
That number answers "did *any* of the S sampled trajectories come close to the
human driver?" — but it is a poor proxy for the question we actually care about
when comparing an *optimized* model (FP8 / NVFP4 / AutoQuant / Torch-TRT /
AOTI) against the vanilla FP16 PyTorch reference:

    "Does the optimized model produce the *same* output as the reference?"

minADE hides three things that matter for optimization QA:
  1. It is a *min* over S samples, so it ignores 5 of the 6 trajectories and
     rewards diversity instead of correctness.
  2. It averages over the horizon, so a large end-of-horizon blow-up is diluted.
  3. It is computed against GT, not against the reference model, so two models
     with the *same average* minADE can disagree clip-by-clip (in our logs the
     per-clip Pearson r between FP16 and FP8 minADE is ~0.03).

This module provides two families of metrics:

  * Family A — ground-truth accuracy (a richer replacement for bare minADE):
    minFDE, meanADE/meanFDE, miss-rate, per-horizon ADE, lateral/longitudinal
    decomposition, heading error, and sample diversity.

  * Family B — fidelity to the FP16 reference (the key signal for optimization):
    reference-ADE, trajectory-set Chamfer distance, and reasoning-text metrics
    (meta-action exact-match, token-level overlap, and embedding cosine
    similarity). Plus logit-level KL / top-1 agreement when logits are captured.

Everything is numpy-based and dependency-light. Optional embedding-cosine uses
``sentence-transformers`` if installed and otherwise falls back to a TF-IDF
cosine so the report can always be produced.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _l2(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """Per-(sample, timestep) L2 distance.

    Args:
        pred_xy: (S, T, 2) predicted XY for S samples.
        gt_xy:   (T, 2) ground-truth XY.
    Returns:
        (S, T) distances.
    """
    return np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)


def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    """Extract planar yaw (radians) from rotation matrices.

    Args:
        rot: (..., 3, 3) rotation matrices.
    Returns:
        (...) yaw angle from the first column (heading) of each matrix.
    """
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Family A: ground-truth accuracy metrics
# ---------------------------------------------------------------------------


@dataclass
class TrajAccuracy:
    """Per-clip ground-truth accuracy metrics (all distances in meters)."""

    min_ade: float
    mean_ade: float          # mean over samples (penalizes bad samples, not just the best)
    min_fde: float           # final-displacement error of the best sample
    mean_fde: float
    miss_rate_2m: float      # 1.0 if minFDE > 2 m else 0.0 (per-clip; average later)
    ade_at: dict[str, float] # ADE of the best sample at fixed horizons, e.g. {"1.0s": ...}
    lateral_err: float       # |lateral| error of the best sample, mean over horizon
    longitudinal_err: float  # |longitudinal| error of the best sample, mean over horizon
    heading_err_deg: float   # mean |yaw| error of the best sample (deg), NaN if no rotation
    sample_spread: float     # mean pairwise L2 between the S endpoints (diversity / mode collapse)


def compute_traj_accuracy(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    *,
    pred_yaw: np.ndarray | None = None,
    gt_yaw: np.ndarray | None = None,
    horizon_fracs: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    miss_threshold_m: float = 2.0,
) -> TrajAccuracy:
    """Compute Family-A accuracy metrics for one clip.

    Args:
        pred_xy: (S, T, 2) predicted trajectories.
        gt_xy:   (T, 2) ground-truth trajectory.
        pred_yaw: optional (S, T) predicted yaw (radians).
        gt_yaw:   optional (T,) ground-truth yaw (radians).
        horizon_fracs: fractions of the horizon at which to report ADE.
        miss_threshold_m: FDE threshold (m) above which the clip is a "miss".
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    S, T, _ = pred_xy.shape

    d = _l2(pred_xy, gt_xy)            # (S, T)
    ade = d.mean(axis=-1)             # (S,)
    fde = d[:, -1]                    # (S,)
    best = int(np.argmin(ade))        # index of the best-ADE sample

    # ADE at fixed horizons (using the best sample) -------------------------
    ade_at: dict[str, float] = {}
    for frac in horizon_fracs:
        t_idx = max(0, min(T - 1, int(round(frac * T)) - 1))
        ade_at[f"{frac:g}xT"] = float(d[best, : t_idx + 1].mean())

    # Lateral / longitudinal decomposition of the best sample ---------------
    # Build a per-step GT heading frame from finite differences of the GT path.
    gt_vel = np.diff(gt_xy, axis=0, prepend=gt_xy[:1])      # (T, 2)
    speed = np.linalg.norm(gt_vel, axis=-1, keepdims=True)
    heading = np.divide(gt_vel, speed, out=np.zeros_like(gt_vel), where=speed > 1e-6)
    normal = np.stack([-heading[:, 1], heading[:, 0]], axis=-1)  # left-normal
    err_vec = pred_xy[best] - gt_xy                          # (T, 2)
    longitudinal = np.abs((err_vec * heading).sum(-1)).mean()
    lateral = np.abs((err_vec * normal).sum(-1)).mean()

    # Heading error ---------------------------------------------------------
    if pred_yaw is not None and gt_yaw is not None:
        pred_yaw = np.asarray(pred_yaw, dtype=np.float64)
        gt_yaw = np.asarray(gt_yaw, dtype=np.float64)
        yaw_err = np.abs(wrap_to_pi(pred_yaw[best] - gt_yaw)).mean()
        heading_err_deg = float(np.degrees(yaw_err))
    else:
        heading_err_deg = float("nan")

    # Sample spread (endpoint diversity) ------------------------------------
    endpoints = pred_xy[:, -1, :]                           # (S, 2)
    if S > 1:
        pair = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=-1)
        spread = float(pair[np.triu_indices(S, k=1)].mean())
    else:
        spread = 0.0

    return TrajAccuracy(
        min_ade=float(ade.min()),
        mean_ade=float(ade.mean()),
        min_fde=float(fde.min()),
        mean_fde=float(fde.mean()),
        miss_rate_2m=float(fde.min() > miss_threshold_m),
        ade_at=ade_at,
        lateral_err=float(lateral),
        longitudinal_err=float(longitudinal),
        heading_err_deg=heading_err_deg,
        sample_spread=spread,
    )


# ---------------------------------------------------------------------------
# Family B: fidelity to the reference (vanilla FP16) model
# ---------------------------------------------------------------------------


def chamfer_set_distance(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two trajectory *sets*.

    Compares the cloud of S reference trajectories against the cloud of S
    optimized trajectories without assuming sample i corresponds to sample i
    (sampling order is not stable across models).

    Args:
        set_a: (Sa, T, 2)
        set_b: (Sb, T, 2)
    Returns:
        mean(min_b d(a, b)) + mean(min_a d(a, b)), where d is mean-over-time L2.
    """
    set_a = np.asarray(set_a, dtype=np.float64)
    set_b = np.asarray(set_b, dtype=np.float64)
    # pairwise mean-over-time L2: (Sa, Sb)
    diff = set_a[:, None, :, :] - set_b[None, :, :, :]
    dmat = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return float(dmat.min(axis=1).mean() + dmat.min(axis=0).mean()) / 2.0


def reference_ade(pred_xy: np.ndarray, ref_xy: np.ndarray) -> float:
    """ADE of the optimized output against the reference model's best sample.

    Treats the reference (FP16) trajectory as pseudo-ground-truth. This is the
    most direct "is the output the same?" number: 0 means the optimized model
    reproduced the reference path exactly.

    Args:
        pred_xy: (S, T, 2) optimized samples.
        ref_xy:  (T, 2) reference trajectory (e.g. reference best-ADE sample).
    """
    d = _l2(np.asarray(pred_xy, np.float64), np.asarray(ref_xy, np.float64))
    return float(d.mean(axis=-1).min())


# --- reasoning-text fidelity ------------------------------------------------


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(a: str, b: str) -> float:
    """1.0 if normalized strings are identical (good for categorical meta_action)."""
    return float(normalize_text(a) == normalize_text(b))


def token_f1(a: str, b: str) -> float:
    """Bag-of-tokens F1 overlap between two strings (order-insensitive)."""
    ta, tb = normalize_text(a).split(), normalize_text(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = 0
    tb_pool = list(tb)
    for tok in ta:
        if tok in tb_pool:
            tb_pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(ta)
    recall = common / len(tb)
    return 2 * precision * recall / (precision + recall)


def char_similarity(a: str, b: str) -> float:
    """Normalized character-level similarity (1 - edit distance ratio)."""
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


class TextEmbedder:
    """Cosine-similarity embedder for reasoning text.

    Uses sentence-transformers when available (semantically meaningful), and
    falls back to a TF-IDF vectorizer so the metric is always computable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.backend = None
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

                self._vec = TfidfVectorizer()
                self.backend = "tfidf"
            except Exception:
                self.backend = "none"

    def cosine(self, a: str, b: str) -> float:
        a, b = normalize_text(a), normalize_text(b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        if self.backend == "sentence-transformers":
            emb = self._model.encode([a, b], normalize_embeddings=True)
            return float(np.dot(emb[0], emb[1]))
        if self.backend == "tfidf":
            try:
                m = self._vec.fit_transform([a, b]).toarray()
            except ValueError:
                return 0.0
            denom = np.linalg.norm(m[0]) * np.linalg.norm(m[1])
            return float(np.dot(m[0], m[1]) / denom) if denom > 0 else 0.0
        return float("nan")


@dataclass
class TextFidelity:
    meta_action_match: float
    answer_match: float
    cot_token_f1: float
    cot_char_sim: float
    cot_cosine: float


def compute_text_fidelity(
    ref_extra: dict,
    opt_extra: dict,
    *,
    embedder: TextEmbedder | None = None,
    sample_idx: int = 0,
) -> TextFidelity:
    """Compare reference vs optimized reasoning text for one clip.

    ``extra`` dicts come from ``sample_trajectories_from_data_with_vlm_rollout(...,
    return_extra=True)`` and contain ``cot``/``meta_action``/``answer`` arrays of
    shape [B, sets, samples].
    """

    def _get(extra: dict, key: str) -> str:
        arr = np.asarray(extra.get(key, np.array([[[""]]])))
        flat = arr.reshape(-1)
        idx = min(sample_idx, len(flat) - 1)
        return str(flat[idx])

    ref_cot, opt_cot = _get(ref_extra, "cot"), _get(opt_extra, "cot")
    cosine = embedder.cosine(ref_cot, opt_cot) if embedder else float("nan")
    return TextFidelity(
        meta_action_match=exact_match(_get(ref_extra, "meta_action"), _get(opt_extra, "meta_action")),
        answer_match=exact_match(_get(ref_extra, "answer"), _get(opt_extra, "answer")),
        cot_token_f1=token_f1(ref_cot, opt_cot),
        cot_char_sim=char_similarity(ref_cot, opt_cot),
        cot_cosine=cosine,
    )


# --- logit-level fidelity (most sensitive quantization probe) ---------------


def logit_fidelity(ref_logits: np.ndarray, opt_logits: np.ndarray) -> dict[str, float]:
    """Per-step distribution agreement between reference and optimized logits.

    Use with greedy / teacher-forced decoding so the two models score the same
    token positions. Catches numerical drift long before it changes the decoded
    text, which makes it the most sensitive fidelity signal for quantization.

    Args:
        ref_logits: (L, V) reference next-token logits.
        opt_logits: (L, V) optimized next-token logits, same positions.
    Returns:
        dict with top-1 agreement rate and mean KL(ref || opt).
    """
    ref_logits = np.asarray(ref_logits, np.float64)
    opt_logits = np.asarray(opt_logits, np.float64)

    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    p = _softmax(ref_logits)
    q = _softmax(opt_logits)
    top1 = (ref_logits.argmax(-1) == opt_logits.argmax(-1)).mean()
    kl = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1).mean()
    return {"top1_agreement": float(top1), "mean_kl": float(kl)}


# ---------------------------------------------------------------------------
# Aggregation across clips
# ---------------------------------------------------------------------------


@dataclass
class MetricAccumulator:
    """Accumulate per-clip metrics and report dataset-level summaries."""

    accuracy: list[TrajAccuracy] = field(default_factory=list)
    ref_ade: list[float] = field(default_factory=list)
    chamfer: list[float] = field(default_factory=list)
    text: list[TextFidelity] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.accuracy:
            out["minADE"] = float(np.mean([a.min_ade for a in self.accuracy]))
            out["meanADE"] = float(np.mean([a.mean_ade for a in self.accuracy]))
            out["minFDE"] = float(np.mean([a.min_fde for a in self.accuracy]))
            out["MissRate@2m"] = float(np.mean([a.miss_rate_2m for a in self.accuracy]))
            out["lateral_err"] = float(np.mean([a.lateral_err for a in self.accuracy]))
            out["longitudinal_err"] = float(np.mean([a.longitudinal_err for a in self.accuracy]))
            out["heading_err_deg"] = float(np.nanmean([a.heading_err_deg for a in self.accuracy]))
            out["sample_spread"] = float(np.mean([a.sample_spread for a in self.accuracy]))
        if self.ref_ade:
            out["referenceADE"] = float(np.mean(self.ref_ade))
        if self.chamfer:
            out["chamfer"] = float(np.mean(self.chamfer))
        if self.text:
            out["meta_action_match"] = float(np.mean([t.meta_action_match for t in self.text]))
            out["answer_match"] = float(np.mean([t.answer_match for t in self.text]))
            out["cot_token_f1"] = float(np.mean([t.cot_token_f1 for t in self.text]))
            out["cot_char_sim"] = float(np.mean([t.cot_char_sim for t in self.text]))
            out["cot_cosine"] = float(np.nanmean([t.cot_cosine for t in self.text]))
        return out


def paired_correlation(ref_vals: Sequence[float], opt_vals: Sequence[float]) -> dict[str, float]:
    """Pearson/Spearman correlation + regression stats of per-clip minADE.

    A high *average* similarity with a *low* correlation is the signature of an
    optimization that preserves dataset-level scores while scrambling per-clip
    behaviour — exactly what bare minADE fails to surface.
    """
    a = np.asarray(ref_vals, np.float64)
    b = np.asarray(opt_vals, np.float64)
    pearson = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    spearman = float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else float("nan")
    diff = b - a
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mean_abs_delta": float(np.abs(diff).mean()),
        "regression_rate@0.2m": float((diff > 0.2).mean()),
        "frac_changed@0.5m": float((np.abs(diff) > 0.5).mean()),
    }
