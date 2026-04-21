# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRT-LLM plugin-based LM compilation for the Alpamayo VLM language model.

Mirrors the file structure of ``alpamayo1_5.trt.lm_with_cache`` but uses the
TRT-LLM ``xqa_attn`` plugin for attention.  The compiled engine consumes
``(inputs_embeds, kv_caches, ctx_len, ds_stack)`` and returns
``(logits, updated_kv_caches)``.
"""

from __future__ import annotations

import copy
import gc
import logging
from contextlib import nullcontext
from typing import Any, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FP16 = torch.float16


class PluginWrapperDSInput(nn.Module):
    """LM forward that uses the plugin self-attention and adds deepstack features per layer."""

    def __init__(self, lm: nn.Module, lm_head: nn.Module, num_ds: int):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.num_ds = int(num_ds)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        new_kvs: list[torch.Tensor] = []
        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                past_key_value=kv_caches[i],
                ctx_len=ctx_len,
            )
            hidden = residual + hidden
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            new_kvs.append(kv)
            if i < self.num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :]
        hidden = self.lm.norm(hidden)
        return self.lm_head(hidden), new_kvs


def _build_rope_cache(
    lm: nn.Module,
    S_input: int,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-compute concatenated ``(cos, sin)`` RoPE cache for all positions up to ``max_seq_len``."""
    with torch.no_grad():
        d_eff = torch.arange(S_input, max_seq_len, device=device).float()
        d_eff = d_eff + rope_deltas.to(device).float().squeeze()
        d_3d = d_eff.view(1, 1, -1).expand(3, 1, -1).long()
        full_pos = torch.cat([position_ids.to(device), d_3d], dim=2)
        cos, sin = lm.rotary_emb(torch.ones(1, device=device, dtype=FP16), full_pos)
        h2 = head_dim // 2
        rope_cache = torch.cat(
            [cos[:, :max_seq_len, :h2].float(), sin[:, :max_seq_len, :h2].float()],
            dim=-1,
        )
    return rope_cache


def _install_plugin_attention(
    lm: nn.Module,
    config: Any,
    rope_cache: torch.Tensor,
) -> None:
    """Replace each layer's ``self_attn`` with a plugin-backed attention module."""
    from plugin_utils import PluginAttention

    for i, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, i, rope_cache)


def _export_plugin_wrapper(
    wrapper: nn.Module,
    example_embeds: torch.Tensor,
    example_kvs: list[torch.Tensor],
    example_ctx: torch.Tensor,
    example_ds: torch.Tensor,
    num_layers: int,
    max_seq_len: int,
) -> "torch.export.ExportedProgram":
    """Export the plugin LM wrapper to ``ExportedProgram`` with dynamic seq-len."""
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    dynamic_shapes = {
        "inputs_embeds": {1: seq_dim},
        "kv_caches": [{}] * num_layers,
        "ctx_len": {},
        "ds_stack": {},
    }
    args = (example_embeds, example_kvs, example_ctx, example_ds)
    try:
        return torch.export.export(
            wrapper, args=args, dynamic_shapes=dynamic_shapes, strict=False,
        )
    except Exception:
        return torch.export._trace._export(
            wrapper, args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def _plugin_lm_smoke_check(
    ref_wrapper: nn.Module,
    trt_lm: nn.Module,
    example_embeds: torch.Tensor,
    example_ctx: torch.Tensor,
    example_ds: torch.Tensor,
    config: Any,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Run one reference-vs-TRT forward pass and log the L2 relative error."""
    from plugin_utils import create_kv_caches

    try:
        with torch.no_grad():
            kvs_ref = create_kv_caches(config, max_seq_len, batch_size, device, FP16)
            kvs_trt = create_kv_caches(config, max_seq_len, batch_size, device, FP16)
            ref_logits, _ = ref_wrapper(example_embeds, kvs_ref, example_ctx, example_ds)
            trt_logits, _ = trt_lm(example_embeds, kvs_trt, example_ctx, example_ds)
        ref_f = ref_logits.float()
        trt_f = trt_logits.float()
        l2 = (ref_f - trt_f).norm() / ref_f.norm().clamp_min(1e-8)
        logger.info("LM plugin TRT smoke-check L2 relative error: %.4f", l2.item())
        seq = int(example_ctx[0])
        ref_kv = torch.stack([k[:, :, :, :seq, :] for k in kvs_ref])
        trt_kv = torch.stack([k[:, :, :, :seq, :] for k in kvs_trt])
        kv_l2 = (ref_kv.float() - trt_kv.float()).norm() / ref_kv.float().norm().clamp_min(1e-8)
        logger.info("LM plugin TRT KV-cache L2 relative error: %.4f", kv_l2.item())
    except Exception as e:
        logger.warning("LM plugin TRT smoke test failed: %s", e)


def compile_vlm_lm_trt_with_plugin(
    model: nn.Module,
    S_input: int,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    num_ds_layers: int,
    max_seq_len: int = 4096,
    batch_size: int = 1,
    device: str = "cuda",
    debug: bool = False,
    accuracy_check: bool = False,
) -> tuple[nn.Module, nn.Embedding, Any]:
    """Compile the VLM's language model as a TRT engine with TRT-LLM attention plugin.

    Parameters match ``compile_vlm_lm_trt_with_cache`` as closely as possible,
    with three extra required arguments that describe the prefill sequence:

    - ``S_input``: prefill sequence length used to build the RoPE cache.
    - ``position_ids``: ``[3, 1, S_input]`` from ``get_rope_index``.
    - ``rope_deltas``: ``[1, 1]`` from ``get_rope_index``.
    - ``num_ds_layers``: number of visual deepstack layers to add at prefill.

    Returns ``(trt_lm, embed_tokens_fp16, lm_config)``.  The compiled engine
    and its metadata are also attached to ``model`` as private attributes so
    downstream inference helpers can find them without an extra dict.
    """
    import torch_tensorrt
    from plugin_utils import (
        create_kv_caches,
        load_plugin,
        register_plugin_op,
        set_plugin_config_from_model,
    )

    dev = torch.device(device)
    logger.info("Compiling VLM LM with TRT-LLM attention plugin (FP16, batch=%d)", batch_size)
    load_plugin()
    register_plugin_op()

    lm_ref = model.vlm.model.language_model
    config = lm_ref.config
    head_dim = config.head_dim
    num_kv_heads = config.num_key_value_heads
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    set_plugin_config_from_model(config, max_seq_len)

    lm = copy.deepcopy(lm_ref).to(dtype=FP16, device=dev).eval()
    rope_cache = _build_rope_cache(
        lm=lm,
        S_input=S_input,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        device=dev,
    )

    embed_tokens = copy.deepcopy(lm.embed_tokens)
    _install_plugin_attention(lm, config, rope_cache)

    lm_head = copy.deepcopy(model.vlm.lm_head).to(dtype=FP16, device=dev).eval()
    wrapper = PluginWrapperDSInput(lm, lm_head, num_ds_layers).to(device=dev).eval()

    B = int(batch_size)
    example_embeds = torch.randn(B, 3, hidden_size, dtype=FP16, device=dev)
    example_ctx = torch.tensor([3] * B, dtype=torch.int32, device=dev)
    example_kvs = [
        torch.zeros(B, 2, num_kv_heads, max_seq_len, head_dim, dtype=FP16, device=dev)
        for _ in range(num_layers)
    ]
    example_ds = torch.zeros(num_ds_layers, B, max_seq_len, hidden_size, dtype=FP16, device=dev)

    ep = _export_plugin_wrapper(
        wrapper=wrapper,
        example_embeds=example_embeds,
        example_kvs=example_kvs,
        example_ctx=example_ctx,
        example_ds=example_ds,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    trt_settings: dict = {
        "enabled_precisions": {torch.float32},
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "disable_tf32": True,
        "min_block_size": 1,
        "dryrun": False,
        "device": dev,
        "use_python_runtime": True,
        "decompose_attention": True,
    }

    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_lm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[example_embeds, example_kvs, example_ctx, example_ds],
            **trt_settings,
        )

    if accuracy_check:
        _plugin_lm_smoke_check(
            ref_wrapper=wrapper,
            trt_lm=trt_lm,
            example_embeds=example_embeds,
            example_ctx=example_ctx,
            example_ds=example_ds,
            config=config,
            max_seq_len=max_seq_len,
            batch_size=B,
            device=dev,
        )

    model._trt_plugin_lm_backbone = trt_lm
    model._trt_plugin_lm_config = config
    model._trt_plugin_lm_max_seq_len = int(max_seq_len)
    model._trt_plugin_lm_batch_size = B
    model._trt_plugin_num_ds_layers = int(num_ds_layers)
    model._trt_plugin_embed_tokens = embed_tokens
    model._trt_plugin_rope_deltas_ref = rope_deltas.detach().clone().to(dev)
    model._trt_plugin_S_input = int(S_input)

    del wrapper, example_embeds, example_kvs, example_ds, ep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("✓ LM plugin TRT compiled (batch=%d, max_seq_len=%d)", B, max_seq_len)
    return trt_lm, embed_tokens, config
