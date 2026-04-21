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

from __future__ import annotations

import contextlib
import copy
import logging
import time
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

logger = logging.getLogger(__name__)


def run_vlm_preprocessing(
    model: nn.Module,
    model_inputs: dict[str, Any],
    trt_vision: nn.Module | None = None,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Embed tokens, run the vision encoder, fuse features, and compute RoPE.

    Returns ``(input_ids, inputs_embeds, deepstack_image_embeds, visual_pos_masks,
    position_ids, rope_deltas)``. When ``trt_vision`` is provided, the VLM's
    ``visual.forward`` is temporarily swapped to use it.
    """
    device = torch.device(device)
    tokenized_data = copy.deepcopy(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )

    vlm_model = model.vlm.model
    lm_ref = vlm_model.language_model

    original_fwd = None
    if trt_vision is not None:
        original_fwd = vlm_model.visual.forward
        vlm_model.visual.forward = trt_vision.forward

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        inputs_embeds = lm_ref.embed_tokens(input_ids)
        pv = tokenized_data["pixel_values"].to(device)
        igt = tokenized_data["image_grid_thw"].to(device)
        image_embeds, ds_embeds = vlm_model.get_image_features(pv, igt)
        image_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vlm_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_cat
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_cat)
        vis_masks = image_mask[..., 0]
        attn = tokenized_data.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        position_ids, rope_deltas = vlm_model.get_rope_index(
            input_ids, igt, video_grid_thw=None, attention_mask=attn
        )

    if original_fwd is not None:
        vlm_model.visual.forward = original_fwd

    del pv, igt, image_embeds, image_cat, image_mask
    torch.cuda.empty_cache()
    return input_ids, inputs_embeds, ds_embeds, vis_masks, position_ids, rope_deltas


def sample_token(
    logits: torch.Tensor,
    traj_token_offset: int,
    traj_vocab_size: int,
    temperature: float = 0.6,
    top_p: float = 0.98,
) -> torch.Tensor:
    """Temperature-scaled top-p (nucleus) sampling with trajectory-token masking."""
    logits = logits[:, -1, :].float()
    logits[:, traj_token_offset : traj_token_offset + traj_vocab_size] = float("-inf")
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    remove_mask = cum_probs - probs >= top_p
    sorted_logits[remove_mask] = float("-inf")
    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def configure_generation(
    model,
    *,
    num_return_sequences: int,
    max_new_tokens: int,
) -> None:
    generation_config = model.vlm.generation_config
    generation_config.top_p = 0.98
    generation_config.temperature = 0.6
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_return_sequences
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id


def compile_vision_trt(
    model: nn.Module,
    model_inputs: dict[str, Any],
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling Vision Encoder with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.vision import compile_vision_model

    trt_vision = compile_vision_model(
        model.vlm.model.visual,
        model_inputs,
        device="cuda",
        debug=False,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_vision is None:
        logger.error("Vision TRT compilation failed")
        return None

    logger.info("✓ Vision encoder compiled with TRT")
    return trt_vision


def compile_diffusion_no_cache_trt(
    model: nn.Module,
    max_prefix_len: int,
    batch_size: int = 1,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling No-Cache Diffusion Step with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.diffusion import compile_diffusion_step_no_cache

    compile_diffusion_step_no_cache(
        model,
        max_prefix_len=max_prefix_len,
        batch_size=int(batch_size),
        device="cuda",
        offload_module_to_cpu=offload_module_to_cpu,
        debug=False,
        accuracy_check=True,
    )
    if not hasattr(model, "_trt_diffusion_step_no_cache"):
        logger.error("No-cache diffusion TRT compilation failed")
        return None

    logger.info("✓ No-cache diffusion step compiled with TRT")
    return model._trt_diffusion_step_no_cache


def compile_language_trt(
    model: nn.Module,
    max_seq_len: int = 4096,
    max_prefix_len: int | None = None,
    batch_size: int = 1,
    offload_module_to_cpu: bool = False,
) -> nn.Module | None:
    logger.info("\n" + "=" * 60)
    logger.info("Compiling Language Model with TensorRT")
    logger.info("=" * 60)

    from alpamayo_r1.trt.lm_with_cache import compile_vlm_lm_trt_with_cache

    compiled_model = compile_vlm_lm_trt_with_cache(
        model,
        max_seq_len=max_seq_len,
        max_prefix_len=max_seq_len if max_prefix_len is None else max_prefix_len,
        batch_size=batch_size,
        device="cuda",
        offload_module_to_cpu=offload_module_to_cpu,
        debug=False,
        accuracy_check=True,
    )
    model._trt_vlm_backbone = compiled_model
    model._trt_lm_max_batch_size = int(batch_size)
    model._trt_lm_batch_size = int(batch_size)
    logger.info("✓ Language model compiled with TRT")
    return model._trt_vlm_backbone


def measure_prefix_seq_len_for_trt(
    model,
    create_inputs_fn: callable,
    seed: int = 42,
    max_generation_length: int = 256,
) -> int:
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import StopAfterEOS, to_special_token

    torch.cuda.manual_seed_all(seed)
    _inputs = create_inputs_fn()
    _tokenized = _inputs["tokenized_data"]
    _input_ids = _tokenized.pop("input_ids")
    _input_ids = model.fuse_traj_tokens(
        _input_ids,
        {
            "ego_history_xyz": _inputs["ego_history_xyz"],
            "ego_history_rot": _inputs["ego_history_rot"],
        },
    )
    _eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

    configure_generation(model, num_return_sequences=1, max_new_tokens=max_generation_length)
    output_embeddings = model.vlm.get_output_embeddings()
    vlm_dtype = (
        output_embeddings.weight.dtype
        if output_embeddings is not None
        else next(model.vlm.parameters()).dtype
    )
    use_autocast = torch.cuda.is_available() and vlm_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast("cuda", dtype=vlm_dtype) if use_autocast else contextlib.nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            _vlm_out = model.vlm.generate(
                input_ids=_input_ids,
                generation_config=model.vlm.generation_config,
                stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=_eos_id)]),
                logits_processor=LogitsProcessorList(
                    [
                        ExpertLogitsProcessor(
                            traj_token_offset=model.config.traj_token_start_idx,
                            traj_vocab_size=model.config.traj_vocab_size,
                        )
                    ]
                ),
                **_tokenized,
            )
    return int(_vlm_out.past_key_values.get_seq_length())


def compile_trt_modules(
    model: nn.Module,
    create_inputs_fn: callable,
    *,
    seed: int = 42,
    offload_module_to_cpu: bool = False,
    max_generation_length: int = 256,
    lm_max_seq_len: int | None = None,
    max_prefix_len: int | None = None,
    num_traj_samples: int = 1,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, int]:
    logger.info("\nMeasuring prefix sequence length for TRT sizing...")
    observed_prefix_seq_len = measure_prefix_seq_len_for_trt(
        model,
        create_inputs_fn,
        seed=seed,
        max_generation_length=max_generation_length,
    )
    prefix_seq_len = observed_prefix_seq_len if max_prefix_len is None else int(max_prefix_len)
    logger.info(f"  observed prefix_seq_len = {observed_prefix_seq_len}")
    logger.info(f"  using max_prefix_len     = {prefix_seq_len}")

    model_inputs = create_inputs_fn()
    trt_vision = None
    trt_vision = compile_vision_trt(
        model,
        model_inputs,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_vision is None:
        logger.error("Failed to compile vision model")

    lm_seq_len = (
        int(prefix_seq_len + max_generation_length)
        if lm_max_seq_len is None
        else int(lm_max_seq_len)
    )
    compile_inputs = create_inputs_fn()
    compile_batch = max(2, int(compile_inputs["ego_history_xyz"].shape[0]) * int(num_traj_samples))
    logger.info(f"  lm compile batch size   = {compile_batch}")

    
    
    trt_lm = None
    trt_lm = compile_language_trt(
        model,
        max_seq_len=lm_seq_len,
        max_prefix_len=lm_seq_len,
        batch_size=compile_batch,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_lm is None:
        logger.error("Failed to compile language model")


    trt_diffusion = None
    diffusion_max_prefix_len = max(int(prefix_seq_len), int(lm_seq_len))
    logger.info(f"  diffusion max_prefix_len = {diffusion_max_prefix_len}")
    trt_diffusion = compile_diffusion_no_cache_trt(
        model,
        max_prefix_len=diffusion_max_prefix_len,
        batch_size=compile_batch,
        offload_module_to_cpu=offload_module_to_cpu,
    )
    if trt_diffusion is None:
        logger.error("Failed to compile no-cache diffusion step")
    else:
        model._trt_diffusion_batch_size = int(compile_batch)


    return trt_vision, trt_lm, trt_diffusion, prefix_seq_len


def run_inference_trt(
    model,
    create_inputs_fn: callable,
    trt_vision: nn.Module | None,
    trt_lm: nn.Module | None,
    trt_diffusion: nn.Module | None,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    """Run inference with TRT vision/language/diffusion modules."""
    from alpamayo_r1.models.alpamayo_r1 import ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import (
        StopAfterEOS,
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )
    from alpamayo_r1.trt.prefix_cache import PrefixKVCache, stack_prefix_kv_from_cache

    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()

    dtype = torch.float16
    device = "cuda"

    start_time = time.perf_counter()

    with torch.autocast("cuda", dtype=dtype):
        ego_history_xyz = model_inputs["ego_history_xyz"]
        ego_history_rot = model_inputs["ego_history_rot"]
        B, _, _, _ = ego_history_xyz.shape
        tokenized_data = model_inputs["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

        original_vision_forward = None
        if trt_vision is not None:
            original_vision_forward = model.vlm.model.visual.forward
            model.vlm.model.visual.forward = trt_vision.forward

        eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        configure_generation(
            model,
            num_return_sequences=num_traj_samples,
            max_new_tokens=max_generation_length,
        )

        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        )

        vlm_outputs = model.vlm.generate(
            input_ids=input_ids,
            generation_config=model.vlm.generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = model.vlm.model.rope_deltas

        if original_vision_forward is not None:
            model.vlm.model.visual.forward = original_vision_forward

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
        )

        b_star = vlm_outputs.sequences.shape[0]
        compiled_max_batch = getattr(
            model,
            "_trt_lm_max_batch_size",
            getattr(model, "_trt_lm_batch_size", None),
        )
        if trt_lm is not None and compiled_max_batch is not None and int(b_star) > int(compiled_max_batch):
            raise ValueError(
                f"TRT LM batch mismatch: compiled max batch={compiled_max_batch}, runtime batch={b_star}. "
                "Recompile TRT LM with a larger num_traj_samples/input batch."
            )
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids = position_ids + delta.to(device)

        neg_inf = torch.finfo(dtype).min
        attention_mask = torch.zeros(
            b_star,
            1,
            n_diffusion_tokens,
            prefill_seq_len + n_diffusion_tokens,
            dtype=dtype,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = neg_inf

        prefix_k, prefix_v = stack_prefix_kv_from_cache(
            prompt_cache,
            device=torch.device(device),
            dtype=dtype,
        )
        # Defensive cast to guarantee TRT KV input dtypes stay aligned.
        prefix_k = prefix_k.to(device=device, dtype=dtype).contiguous()
        prefix_v = prefix_v.to(device=device, dtype=dtype).contiguous()
        prompt_cache = PrefixKVCache(prefix_k, prefix_v)
        if trt_diffusion is not None:
            logger.info(f"  prefix_k shape: {prefix_k.shape}")
            if prefix_k.dtype != dtype or prefix_v.dtype != dtype:
                raise TypeError(
                    f"TRT KV dtype mismatch: expected {dtype}, got "
                    f"prefix_k={prefix_k.dtype}, prefix_v={prefix_v.dtype}"
                )
            compiled_diff_batch = getattr(model, "_trt_diffusion_batch_size", None)
            if compiled_diff_batch is not None and int(b_star) != int(compiled_diff_batch):
                raise ValueError(
                    f"TRT diffusion batch mismatch: compiled batch={compiled_diff_batch}, runtime batch={b_star}. "
                    "Recompile TRT diffusion with matching batch (input_batch * num_traj_samples)."
                )

        forward_kwargs = {}
        if model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        if trt_diffusion is not None:

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return trt_diffusion(
                    x.to(dtype), t.to(dtype), prefix_k, prefix_v, position_ids, attention_mask
                )
        else:

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                b = x.shape[0]
                future_token_embeds = model.action_in_proj(x, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(b, n_diffusion_tokens, -1)
                expert_out = model.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=position_ids,
                    past_key_values=prompt_cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **forward_kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
                return model.action_out_proj(last_hidden).view(
                    -1, *model.action_space.get_action_space_dims()
                )

        total_batch = B * num_traj_samples
        sampled_action = model.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )

        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_traj_samples
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_traj_samples
        )
        pred_xyz, pred_rot = model.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples
        )

        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        for k in extra:
            extra[k] = np.array(extra[k]).reshape([input_ids.shape[0], 1, num_traj_samples])

    inference_time = time.perf_counter() - start_time
    return pred_xyz, pred_rot, extra, inference_time


def compile_trt_with_plugin(
    model: nn.Module,
    create_inputs_fn: callable,
    *,
    seed: int = 42,
    offload_module_to_cpu: bool = False,
    max_generation_length: int = 256,
    num_traj_samples: int = 1,
    max_seq_len: int | None = None,
    debug: bool = False,
    accuracy_check: bool = True,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, dict]:
    """Compile (Vision, Plugin LM, Diffusion) TRT engines for plugin-based inference.

    Mirrors :func:`compile_trt_modules` but swaps ``compile_vlm_lm_trt_with_cache``
    for :func:`compile_vlm_lm_trt_with_plugin` (TRT-LLM ``xqa_attn`` plugin).

    Returns ``(trt_vision, trt_lm, trt_diffusion, plugin_info)`` where
    ``plugin_info`` carries metadata the plugin decode loop needs:

        - ``embed_tokens``  : FP16 token-embedding module
        - ``lm_config``     : HF text config
        - ``S_input``       : prefill sequence length used to build RoPE
        - ``max_seq_len``   : compiled KV capacity
        - ``num_ds_layers`` : number of deepstack layers
        - ``rope_deltas_ref``: reference RoPE delta tensor (copied to CUDA)
    """
    from alpamayo_r1.trt.diffusion import compile_diffusion_fp16
    from alpamayo_r1.trt.lm_plugin import compile_vlm_lm_trt_with_plugin
    from alpamayo_r1.trt.vision import compile_vision_fp16

    torch.cuda.manual_seed_all(seed)

    logger.info("\n" + "=" * 60)
    logger.info("Compiling Plugin TRT pipeline (Vision + Plugin LM + Diffusion)")
    logger.info("=" * 60)

    model_inputs = create_inputs_fn()

    _, embeds, ds_embeds, _, position_ids, rope_deltas = run_vlm_preprocessing(
        model, model_inputs, trt_vision=None,
    )
    S_input = int(embeds.shape[1])
    num_ds_layers = len(ds_embeds)
    if max_seq_len is None:
        lm_max_seq_len = max(4096, S_input + max_generation_length + 100)
    else:
        lm_max_seq_len = int(max_seq_len)
    logger.info(f"  S_input={S_input}, num_ds_layers={num_ds_layers}, max_seq_len={lm_max_seq_len}")

    batch_size = max(1, int(model_inputs["ego_history_xyz"].shape[0]) * int(num_traj_samples))
    logger.info(f"  plugin compile batch size = {batch_size}")

    logger.info("\n" + "=" * 60)
    logger.info("Compiling Vision Encoder (FP16 plugin pipeline)")
    logger.info("=" * 60)
    trt_vision = compile_vision_fp16(model, model_inputs)
    if trt_vision is None:
        logger.error("Vision TRT compilation failed")

    logger.info("\n" + "=" * 60)
    logger.info("Compiling Plugin Language Model")
    logger.info("=" * 60)
    trt_lm, _embed_tokens, _lm_config = compile_vlm_lm_trt_with_plugin(
        model,
        S_input=S_input,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        num_ds_layers=num_ds_layers,
        max_seq_len=lm_max_seq_len,
        batch_size=batch_size,
        device="cuda",
        debug=debug,
        accuracy_check=accuracy_check,
    )
    if trt_lm is None:
        logger.error("Plugin LM TRT compilation failed")

    logger.info("\n" + "=" * 60)
    logger.info("Compiling Diffusion Step (FP16 plugin pipeline)")
    logger.info("=" * 60)
    trt_diffusion = compile_diffusion_fp16(
        model,
        min_prefix_len=S_input,
        max_prefix_len=S_input + max_generation_length + 10,
        batch_size=batch_size,
    )
    if trt_diffusion is not None:
        model._trt_diffusion_batch_size = int(batch_size)
    else:
        logger.error("Diffusion TRT compilation failed")

    plugin_info = {
        "embed_tokens": _embed_tokens,
        "lm_config": _lm_config,
        "S_input": S_input,
        "max_seq_len": lm_max_seq_len,
        "num_ds_layers": num_ds_layers,
        "rope_deltas_ref": rope_deltas.detach().clone(),
        "batch_size": batch_size,
    }

    del embeds, ds_embeds
    torch.cuda.empty_cache()

    logger.info("✓ Plugin TRT pipeline compiled")
    return trt_vision, trt_lm, trt_diffusion, plugin_info


def run_inference_trt_plugin(
    model,
    create_inputs_fn: callable,
    trt_vision: nn.Module | None,
    trt_lm: nn.Module | None,
    trt_diffusion: nn.Module | None,
    plugin_info: dict,
    seed: int = 42,
    num_traj_samples: int = 1,
    max_generation_length: int = 256,
    top_p: float = 0.98,
    temperature: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, dict, float]:
    """Run inference with the plugin TRT pipeline (Vision+Plugin LM+Diffusion).

    Mirrors :func:`run_inference_trt` but drives decode via a manual token
    loop that feeds the TRT-LLM ``xqa_attn`` plugin's in-place KV cache.
    """
    from alpamayo_r1.models.token_utils import (
        extract_text_tokens,
        replace_padding_after_eos,
        to_special_token,
    )
    from plugin_utils import create_kv_caches

    assert trt_lm is not None, "run_inference_trt_plugin requires a compiled plugin LM"

    dtype = torch.float16
    device = torch.device("cuda")

    embed_tokens = plugin_info["embed_tokens"]
    lm_cfg = plugin_info["lm_config"]
    S_input = int(plugin_info["S_input"])
    max_seq_len = int(plugin_info["max_seq_len"])
    num_ds_layers = int(plugin_info["num_ds_layers"])
    rope_deltas_ref = plugin_info["rope_deltas_ref"].to(device)
    hidden_size = lm_cfg.hidden_size

    tokenizer = model.tokenizer
    eos_id = tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    traj_off = model.config.traj_token_start_idx
    traj_vocab = model.config.traj_vocab_size

    torch.cuda.manual_seed_all(seed)
    model_inputs = create_inputs_fn()
    input_batch = int(model_inputs["ego_history_xyz"].shape[0])
    total_bsz = input_batch * int(num_traj_samples)

    start_time = time.perf_counter()

    input_ids, embeds, ds_embeds, vis_masks, _, _ = run_vlm_preprocessing(
        model, model_inputs, trt_vision=trt_vision,
    )

    ds_stack = torch.zeros(num_ds_layers, total_bsz, max_seq_len, hidden_size, dtype=dtype, device=device)
    vis_mask_row = vis_masks
    while vis_mask_row.dim() > 2:
        vis_mask_row = vis_mask_row[..., 0]
    vp = vis_mask_row[0].nonzero(as_tuple=True)[0] if vis_mask_row.dim() == 2 else vis_mask_row.nonzero(as_tuple=True)[0]
    for i in range(min(num_ds_layers, len(ds_embeds))):
        de = ds_embeds[i].to(dtype)
        if de.dim() > 2:
            de = de.squeeze(0)
        if de.numel() > 0 and vp.numel() > 0:
            ds_stack[i, :, vp, :] = de.unsqueeze(0).expand(total_bsz, -1, -1)

    e_batch = embeds.to(dtype).repeat_interleave(num_traj_samples, dim=0)
    kvs = create_kv_caches(lm_cfg, max_seq_len, total_bsz, device, dtype)
    ctx = torch.full((total_bsz,), S_input, dtype=torch.int32, device=device)
    logits, _ = trt_lm(e_batch, kvs, ctx, ds_stack)

    next_token = sample_token(
        logits, traj_off, traj_vocab, temperature=temperature, top_p=top_p,
    )
    generated: list[list[int]] = [[int(t.item())] for t in next_token]
    seen_eos = next_token.squeeze(-1) == eos_id
    done = torch.zeros(total_bsz, dtype=torch.bool, device=device)
    pos = int(S_input)

    embed_tokens = embed_tokens.to(device)
    for _ in range(max_generation_length - 1):
        if bool(done.all()):
            break
        step_embeds = embed_tokens(next_token.squeeze(-1)).to(dtype).unsqueeze(1)
        ctx_step = torch.full((total_bsz,), pos + 1, dtype=torch.int32, device=device)
        logits, _ = trt_lm(step_embeds, kvs, ctx_step, ds_stack)
        next_token = sample_token(
            logits, traj_off, traj_vocab, temperature=temperature, top_p=top_p,
        )
        newly_done = seen_eos & ~done
        done = done | newly_done
        for i in range(total_bsz):
            if not bool(done[i]):
                generated[i].append(int(next_token[i].item()))
        seen_eos = seen_eos | (next_token.squeeze(-1) == eos_id)
        pos += 1

    max_gen_len = max(len(x) for x in generated) if generated else 0
    gen_tokens = torch.full(
        (total_bsz, max_gen_len), tokenizer.pad_token_id, dtype=torch.long, device=device,
    )
    for i, ids in enumerate(generated):
        if len(ids) > 0:
            gen_tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

    input_ids_rep = input_ids.to(device).repeat_interleave(num_traj_samples, dim=0)
    full_sequences = replace_padding_after_eos(
        torch.cat([input_ids_rep, gen_tokens], dim=1),
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    traj_future_start_mask = full_sequences == eos_id
    has_tfs = traj_future_start_mask.any(dim=1)
    tfs_positions = traj_future_start_mask.int().argmax(dim=1)
    last_token_positions = torch.full((total_bsz,), full_sequences.shape[1] - 1, device=device)
    valid_token_pos = torch.where(has_tfs, tfs_positions, last_token_positions)
    offset = valid_token_pos + 1

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    rope_rep = rope_deltas_ref.repeat_interleave(num_traj_samples, dim=0)
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=total_bsz).clone()
    position_ids = position_ids + (rope_rep + offset[:, None]).to(device)

    kv_len = int(pos)
    prefix_k = torch.stack([k[:, 0, :, :kv_len, :] for k in kvs], dim=0)
    prefix_v = torch.stack([k[:, 1, :, :kv_len, :] for k in kvs], dim=0)
    neg_inf = torch.finfo(dtype).min
    attention_mask = torch.zeros(
        total_bsz, 1, n_diffusion_tokens, kv_len + n_diffusion_tokens, dtype=dtype, device=device,
    )
    for i in range(total_bsz):
        attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = neg_inf

    if trt_diffusion is not None:
        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return trt_diffusion(
                x.to(dtype), t.to(dtype), prefix_k, prefix_v, position_ids, attention_mask,
            )
    else:
        from alpamayo_r1.trt.prefix_cache import PrefixKVCache
        prompt_cache = PrefixKVCache(prefix_k, prefix_v)
        prefill_seq_len = prompt_cache.get_seq_length()
        forward_kwargs = {}
        if model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            future_token_embeds = model.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(bsz, n_diffusion_tokens, -1)
            expert_out = model.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            return model.action_out_proj(last_hidden).view(
                -1, *model.action_space.get_action_space_dims()
            )

    sampled_action = model.diffusion.sample(
        batch_size=total_bsz,
        step_fn=step_fn,
        device=device,
        return_all_steps=False,
    )

    hist_xyz = model_inputs["ego_history_xyz"]
    hist_rot = model_inputs["ego_history_rot"]
    hist_xyz_rep = einops.repeat(hist_xyz[:, -1].to(device), "b ... -> (b n) ...", n=num_traj_samples)
    hist_rot_rep = einops.repeat(hist_rot[:, -1].to(device), "b ... -> (b n) ...", n=num_traj_samples)
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)
    pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples)
    pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=num_traj_samples)

    extra = extract_text_tokens(tokenizer, full_sequences)
    for k in extra:
        extra[k] = np.array(extra[k]).reshape([input_batch, 1, num_traj_samples])

    del kvs, prefix_k, prefix_v, ds_stack
    torch.cuda.empty_cache()

    inference_time = time.perf_counter() - start_time
    return pred_xyz, pred_rot, extra, inference_time
