# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace-native static-KV-cache language model for Torch-TensorRT.

This replaces the bespoke ``PrefixKVCache`` decode machinery (``lm_with_cache.py``)
with the HF ``StaticCache`` pattern used by
``transformers.integrations.executorch.TorchExportableModuleForDecoderOnlyLM``:

* The KV cache is a *fixed-size* set of module buffers (``key_cache_i`` /
  ``value_cache_i``) that the model mutates in place at ``cache_position`` via
  ``index_copy_`` (``StaticLayer.update``). The cache disappears from the graph
  I/O, so the decode step is ``torch.export``-able.
* On Torch-TensorRT compile, the fork's buffer-lifting
  (``lift_mutated_buffers`` / ``inline_lifted_buffers_into_gm`` in
  ``torch_tensorrt.dynamo.lowering._buffer_lifting``, wired into
  ``dynamo.compile``) turns those mutated buffers into engine input bindings and
  the ``index_copy``/``slice_scatter`` converter emits ``IKVCacheUpdateLayer``
  with *aliased outputs*: the C++ runtime writes K/V straight into cache storage,
  no per-step copy.
* Attention is left intact (``decompose_attention=False``) so SDPA lowers to a
  TensorRT ``IAttentionLayer`` instead of matmul+softmax.

Qwen3-VL specifics the stock HF wrapper does not handle:

* **M-RoPE** — Qwen3-VL uses 3D rotary positions; we thread ``position_ids``
  (shape ``[3, B, S]``) exactly as ``Qwen3VLModel.compute_3d_position_ids`` does.
* **DeepStack** — visual features are injected at early decoder layers using
  boolean-mask indexing (``Qwen3VLTextModel._deepstack_process``), which is not
  static-shape exportable. DeepStack only matters during *prefill*, so prefill is
  run eagerly through the native model (filling the cache) and only the
  single-token *decode* step (no DeepStack) is exported and TRT-compiled.

The eager prefill and the TRT decode engine share the *same* cache storage: after
compile we rebind a ``StaticCache`` onto the engine's lifted buffers
(:func:`bind_static_cache_to_module`), so prefill writes land in the buffers the
decode engine aliases.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import StaticCache

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _disable_gqa_in_sdpa():
    """Force the SDPA path to materialize repeated KV heads during export.

    Mirrors ``lm_with_cache._export_wrapper``: the GQA fast path in
    ``transformers.integrations.sdpa_attention`` uses ``enable_gqa=True`` on
    ``scaled_dot_product_attention``, which the TensorRT attention converter does
    not support. Disabling it makes the graph repeat KV heads explicitly so SDPA
    lowers cleanly to ``IAttentionLayer``.
    """
    import transformers.integrations.sdpa_attention as _sdpa_mod

    orig = _sdpa_mod.use_gqa_in_sdpa
    _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False
    try:
        yield
    finally:
        _sdpa_mod.use_gqa_in_sdpa = orig


def _head_shapes(text_cfg) -> tuple[int, int]:
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    return int(num_kv_heads), int(head_dim)


class Qwen3VLStaticCacheLM(nn.Module):
    """Export-friendly wrapper around ``Qwen3VLTextModel`` with a static KV cache.

    ``forward(inputs_embeds, cache_position, position_ids, ...) -> last_hidden_state``

    The cache lives as fixed-size module buffers; the wrapped text model mutates
    them in place at ``cache_position``. Decode export feeds ``seq_len == 1`` and
    no DeepStack; prefill (run eagerly) feeds the full prompt plus DeepStack.
    """

    def __init__(
        self,
        language_model: nn.Module,
        *,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.model = language_model
        text_cfg = language_model.config.get_text_config()
        self.max_cache_len = int(max_cache_len)

        self.cache = StaticCache(config=text_cfg, max_cache_len=int(max_cache_len))
        num_kv_heads, head_dim = _head_shapes(text_cfg)
        # Allocate all layers up-front (export cannot trace lazy init).
        self.cache.early_initialization(
            int(batch_size), num_kv_heads, head_dim, dtype, device
        )

        # Register the cache tensors as buffers so torch.export records their
        # in-place mutation as a BUFFER_MUTATION (the hook the aliasing fast path
        # keys off). These are the *same* tensor objects the StaticCache layers
        # mutate, so registering them is enough.
        for i, layer in enumerate(self.cache.layers):
            self.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.register_buffer(f"value_cache_{i}", layer.values, persistent=False)
            self.register_buffer(
                f"cumulative_length_{i}", layer.cumulative_length, persistent=False
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Reset each layer's write cursor to the requested cache position. This is
        # how HF's static-cache wrapper makes a single exported program reusable
        # across decode steps: the StaticLayer.update writes at `cumulative_length`
        # and then increments it.
        for layer in self.cache.layers:
            layer.cumulative_length.copy_(cache_position[0:1])

        out = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return out.last_hidden_state


def cache_buffer_names(module: nn.Module) -> tuple[list[str], list[str], list[str]]:
    """Return sorted ``(key, value, cumulative)`` buffer names on a (possibly
    compiled) module, matched by substring so it is robust to FX name prefixes."""

    def _sorted(substr: str) -> list[str]:
        names = [n for n, _ in module.named_buffers() if substr in n]
        # Sort by the trailing integer so layer order is preserved.
        def _idx(n: str) -> int:
            digits = "".join(c for c in n.split(substr)[-1] if c.isdigit())
            return int(digits) if digits else 0

        return sorted(names, key=_idx)

    return _sorted("key_cache"), _sorted("value_cache"), _sorted("cumulative_length")


def reset_static_cache(module: nn.Module) -> None:
    """Zero a static-cache module's K/V buffers and position counters so a fresh
    generation starts from cache position 0 (mirrors the gemma3 reference)."""
    with torch.no_grad():
        for name, buf in module.named_buffers():
            if "cache" in name or "cumulative_length" in name:
                buf.zero_()


def bind_static_cache_to_module(static_cache: StaticCache, module: nn.Module) -> None:
    """Point a ``StaticCache``'s layer tensors at ``module``'s lifted cache buffers.

    After ``torch_tensorrt.dynamo.compile`` runs ``inline_lifted_buffers_into_gm``,
    the compiled module owns *cloned* cache buffers (separate storage from the
    eager module used for export). To run eager prefill into the same storage the
    decode engine aliases, we rebind a StaticCache's per-layer ``keys`` / ``values``
    / ``cumulative_length`` to the compiled module's buffers.
    """
    key_names, val_names, cum_names = cache_buffer_names(module)
    if not (len(key_names) == len(val_names) == len(static_cache.layers)):
        raise ValueError(
            "Cannot bind StaticCache to module: layer/buffer count mismatch "
            f"(layers={len(static_cache.layers)}, keys={len(key_names)}, values={len(val_names)})"
        )
    bufs = dict(module.named_buffers())
    for i, layer in enumerate(static_cache.layers):
        layer.keys = bufs[key_names[i]]
        layer.values = bufs[val_names[i]]
        if i < len(cum_names):
            layer.cumulative_length = bufs[cum_names[i]]
        layer.is_initialized = True
        layer.dtype = layer.keys.dtype
        layer.device = layer.keys.device
        layer.max_batch_size = layer.keys.shape[0]
        layer.num_heads = layer.keys.shape[1]
        layer.k_head_dim = layer.keys.shape[-1]
        layer.v_head_dim = layer.values.shape[-1]


def stack_static_kv(
    cache: StaticCache, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the filled portion of a ``StaticCache`` as stacked ``[L, B, H, S, D]``.

    Used to hand the prompt KV off to the diffusion expert (which consumes
    ``prefix_k`` / ``prefix_v`` directly).
    """
    k = torch.stack(
        [layer.keys[:, :, :seq_len, :].contiguous() for layer in cache.layers], dim=0
    )
    v = torch.stack(
        [layer.values[:, :, :seq_len, :].contiguous() for layer in cache.layers], dim=0
    )
    return k, v


def export_static_cache_decode(
    module: Qwen3VLStaticCacheLM,
    *,
    hidden_size: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int | None = None,
    strict: bool = False,
) -> "torch.export.ExportedProgram":
    """Export the single-token decode step (``seq_len == 1``, no DeepStack)."""
    example_embeds = torch.zeros(batch_size, 1, hidden_size, dtype=dtype, device=device)
    example_cache_position = torch.zeros(1, dtype=torch.long, device=device)
    example_position_ids = torch.zeros(3, batch_size, 1, dtype=torch.long, device=device)

    dynamic_shapes = None
    if max_batch_size is not None and max_batch_size > 1:
        batch_dim = torch.export.Dim("batch", min=1, max=int(max_batch_size))
        dynamic_shapes = {
            "inputs_embeds": {0: batch_dim},
            "cache_position": None,
            "position_ids": {1: batch_dim},
        }

    with torch.no_grad(), _disable_gqa_in_sdpa():
        kwargs = dict(
            inputs_embeds=example_embeds,
            cache_position=example_cache_position,
            position_ids=example_position_ids,
        )
        try:
            ep = torch.export.export(
                module, args=(), kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=strict
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("torch.export.export(decode) failed (%s); trace fallback", e)
            ep = torch.export._trace._export(
                module,
                args=(),
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
    return ep


def compile_static_cache_lm_trt(
    ep: "torch.export.ExportedProgram",
    *,
    example_inputs: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    debug: bool = False,
    require_aliasing: bool = True,
) -> nn.Module:
    """Compile the decode program; verify the KV cache became aliased engine I/O."""
    import torch_tensorrt

    enabled = {torch.float16} if dtype == torch.float16 else {torch.float32}
    trt_settings: dict[str, Any] = {
        "arg_inputs": example_inputs,
        "enabled_precisions": enabled,
        "use_fp32_acc": True,
        "use_explicit_typing": True,
        "truncate_double": True,
        "decompose_attention": False,  # SDPA -> IAttentionLayer
        "min_block_size": 1,
        "use_python_runtime": False,  # C++ runtime required for aliased KV writes
        "offload_module_to_cpu": False,
        "device": device,
    }
    trt_module = torch_tensorrt.dynamo.compile(ep, **trt_settings)

    aliased = _collect_aliased_io(trt_module)
    if aliased:
        logger.info("Static-cache decode engine: %d aliased KV binding(s)", len(aliased))
        for out_name, (in_name, kind) in list(aliased.items())[:4]:
            logger.info("  %s <-aliased-> %s (kind=%s)", out_name, in_name, kind)
    elif require_aliasing:
        raise RuntimeError(
            "No aliased KV-cache I/O detected on the compiled decode engine. The "
            "static cache will not update in place. Check that the C++ runtime is "
            "in use (use_python_runtime=False) and that buffer lifting is enabled."
        )
    else:
        logger.warning("No aliased KV-cache I/O detected (require_aliasing=False)")
    return trt_module


def _collect_aliased_io(trt_module: nn.Module) -> dict:
    for _, mod in trt_module.named_modules():
        if getattr(mod, "aliased_io", None):
            return dict(mod.aliased_io)
    return {}


def decode_position_ids(
    rope_deltas: torch.Tensor, position: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """3D M-RoPE position ids for a single decode token at absolute ``position``.

    Replicates ``Qwen3VLModel.compute_3d_position_ids`` for the incremental
    (``past_key_values_length > 0``) branch: ``arange(pos, pos+1)`` broadcast over
    the 3 rope dims, offset by the cached ``rope_deltas``.
    """
    base = torch.full((3, batch_size, 1), float(position), dtype=torch.long, device=device)
    delta = rope_deltas.to(device=device, dtype=torch.long).reshape(1, batch_size, 1)
    return base + delta


class StaticCacheLMRunner(nn.Module):
    """Drives prefill (eager, native model) + decode (TRT aliased engine).

    Both phases share one static KV cache: ``self.trt_decode`` owns the lifted
    cache buffers, and ``self.prefill_cache`` is rebound onto those same buffers so
    eager prefill writes land where the decode engine reads/writes.
    """

    def __init__(
        self,
        language_model: nn.Module,
        *,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
        trt_decode: nn.Module,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.max_cache_len = int(max_cache_len)
        self.trt_decode = trt_decode

        text_cfg = language_model.config.get_text_config()
        self.prefill_cache = StaticCache(config=text_cfg, max_cache_len=int(max_cache_len))
        num_kv_heads, head_dim = _head_shapes(text_cfg)
        self.prefill_cache.early_initialization(
            int(batch_size), num_kv_heads, head_dim, dtype, self.device
        )
        # Rebind onto the engine's lifted buffers so prefill + decode share storage.
        bind_static_cache_to_module(self.prefill_cache, trt_decode)

        self.position = 0

    def reset(self) -> None:
        reset_static_cache(self.trt_decode)
        self.position = 0

    @torch.inference_mode()
    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the native multimodal text model over the prompt; fills the cache.

        Returns the prompt ``last_hidden_state`` ``[B, S, H]``.
        """
        self.reset()
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=self.device, dtype=torch.long)
        out = self.language_model(
            inputs_embeds=inputs_embeds.to(self.device, self.dtype),
            position_ids=position_ids.to(self.device),
            attention_mask=None,
            past_key_values=self.prefill_cache,
            cache_position=cache_position,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        self.position = int(seq_len)
        return out.last_hidden_state

    @torch.inference_mode()
    def decode_step(
        self, token_embeds: torch.Tensor, rope_deltas: torch.Tensor
    ) -> torch.Tensor:
        """One decode token through the TRT aliased engine. Returns ``[B, 1, H]``."""
        cache_position = torch.tensor([self.position], dtype=torch.long, device=self.device)
        position_ids = decode_position_ids(
            rope_deltas, self.position, self.batch_size, self.device
        )
        out = self.trt_decode(
            token_embeds.to(self.device, self.dtype), cache_position, position_ids
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.position += 1
        return out

    def stack_kv(self, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Stacked ``[L, B, H, S, D]`` view of the filled cache for the diffusion expert."""
        return stack_static_kv(self.prefill_cache, self.position if seq_len is None else seq_len)


def compile_vlm_lm_trt_static_cache(
    model: nn.Module,
    *,
    max_seq_len: int = 4096,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    debug: bool = False,
    require_aliasing: bool = True,
) -> StaticCacheLMRunner:
    """Compile the Qwen3-VL decode step to a static-cache aliased TRT engine.

    Replaces ``lm_with_cache.compile_vlm_lm_trt_with_cache``. Stores the runner on
    ``model._trt_lm_static_runner`` and returns it.
    """
    dev = torch.device(device)
    language_model = model.vlm.model.language_model

    # Native attention path -> SDPA -> IAttentionLayer.
    language_model.config._attn_implementation = "sdpa"
    for layer in language_model.layers:
        if hasattr(layer.self_attn, "_attn_implementation"):
            layer.self_attn._attn_implementation = "sdpa"
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"

    # Align the (shared) text model with the compiled engine precision. The TRT
    # decode path no longer uses ``model.vlm.generate``, so casting in place is
    # safe; ``embed_tokens`` lives here too and must match the cache dtype.
    language_model.to(device=dev, dtype=dtype)

    text_cfg = language_model.config.get_text_config()
    hidden_size = int(text_cfg.hidden_size)
    bsz = int(batch_size)

    logger.info("=" * 65)
    logger.info("Compiling VLM LM (HF StaticCache + aliased KV I/O) with TRT")
    logger.info("  max_cache_len=%d batch_size=%d dtype=%s", max_seq_len, bsz, dtype)
    logger.info("=" * 65)

    export_module = Qwen3VLStaticCacheLM(
        language_model,
        batch_size=bsz,
        max_cache_len=max_seq_len,
        device=dev,
        dtype=dtype,
    ).eval()

    ep = export_static_cache_decode(
        export_module,
        hidden_size=hidden_size,
        batch_size=bsz,
        device=dev,
        dtype=dtype,
        max_batch_size=bsz,
    )

    example_inputs = [
        torch.zeros(bsz, 1, hidden_size, dtype=dtype, device=dev),
        torch.zeros(1, dtype=torch.long, device=dev),
        torch.zeros(3, bsz, 1, dtype=torch.long, device=dev),
    ]
    trt_decode = compile_static_cache_lm_trt(
        ep,
        example_inputs=example_inputs,
        device=dev,
        dtype=dtype,
        debug=debug,
        require_aliasing=require_aliasing,
    )

    runner = StaticCacheLMRunner(
        language_model,
        batch_size=bsz,
        max_cache_len=max_seq_len,
        device=dev,
        dtype=dtype,
        trt_decode=trt_decode,
    )
    model._trt_lm_static_runner = runner
    model._trt_lm_batch_size = bsz
    model._trt_lm_max_batch_size = bsz
    logger.info("\u2713 VLM LM compiled (static cache, aliased KV); runner installed")
    return runner
