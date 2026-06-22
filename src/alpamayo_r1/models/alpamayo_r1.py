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

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The modified scores tensor with trajectory tokens masked out (set to -inf).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


class AlpamayoR1(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        # `sample_trajectories_from_data_with_vlm_rollout` calls the expert
        # with a 4D *float* additive attention mask (shape
        # ``[bsz, 1, n_diff_tokens, prefix_len + n_diff_tokens]`` with
        # ``finfo.min`` in padded slots).  FlashAttention-2 only consumes a
        # 2D ``[batch, seq]`` bool/int mask; if it sees a 4D float tensor it
        # mis-slices it, calls ``nonzero`` on every non-zero element
        # (including ``-inf``), and the resulting OOB indices trigger a
        # device-side assert deep inside the flash-attn varlen kernel.
        # Force SDPA, which handles arbitrary 4D float masks natively (and
        # still dispatches to Flash-Attention-v2 kernels under the hood on
        # modern GPUs).  This mirrors `alpamayo1_5.trt.diffusion`.
        # Honour an explicit user choice in ``expert_cfg`` if they made one.
        if not (config.expert_cfg and "_attn_implementation" in config.expert_cfg):
            expert_config._attn_implementation = "sdpa"
        self.expert = AutoModel.from_config(expert_config)
        # Some HF versions read `_attn_implementation` off the module rather
        # than its config at first forward; mirror the value to be safe.
        if hasattr(self.expert, "_attn_implementation"):
            self.expert._attn_implementation = expert_config._attn_implementation
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    def teacher_forced_flow_loss_forward(
        self,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Differentiable forward that returns the flow-matching training targets.

        Bypasses autoregressive reasoning generation and diffusion sampling.
        The VLM runs in a single non-sampling forward pass (with ``<traj_future_start>``
        appended to the prompt) to build the prompt KV cache; the expert then runs once
        on a linearly-interpolated noisy action and returns the predicted velocity field.

        Used by ModelOpt AutoQuantize (see ``trt.quantize_utils.auto_quantize_model``)
        to score per-layer quantization choices with a differentiable loss.

        Args:
            data: dict with ``tokenized_data`` (input_ids + other processor outputs),
                ``ego_history_xyz``, ``ego_history_rot``, ``ego_future_xyz``,
                ``ego_future_rot``.

        Returns:
            dict with keys ``v_pred`` and ``v_target``, both shape
            ``(B, n_diffusion_tokens, action_dim)``. Callers compute MSE between them.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        ego_future_xyz = data["ego_future_xyz"]
        ego_future_rot = data["ego_future_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported."

        tokenized_data = dict(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # Append <traj_future_start> so the expert attends through the full prompt
        # that inference would have generated up to the action block.
        traj_future_start_id = self.tokenizer.convert_tokens_to_ids(
            to_special_token("traj_future_start")
        )
        start_col = torch.full(
            (input_ids.shape[0], 1),
            traj_future_start_id,
            dtype=input_ids.dtype,
            device=device,
        )
        input_ids = torch.cat([input_ids, start_col], dim=1)
        if "attention_mask" in tokenized_data and tokenized_data["attention_mask"] is not None:
            am = tokenized_data["attention_mask"]
            tokenized_data["attention_mask"] = torch.cat(
                [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1
            )
        # transformers>=5.x Qwen3-VL builds M-RoPE token types from the processor's
        # ``mm_token_type_ids`` and indexes them by ``attention_mask`` in
        # ``get_rope_index``.  The appended <traj_future_start> is a text token, so
        # extend ``mm_token_type_ids`` by one ``0`` to keep it aligned with
        # ``input_ids``/``attention_mask`` (otherwise the lengths mismatch and
        # indexing raises an IndexError).
        if (
            "mm_token_type_ids" in tokenized_data
            and tokenized_data["mm_token_type_ids"] is not None
        ):
            mmt = tokenized_data["mm_token_type_ids"]
            tokenized_data["mm_token_type_ids"] = torch.cat(
                [mmt, torch.zeros((mmt.shape[0], 1), dtype=mmt.dtype, device=mmt.device)],
                dim=1,
            )

        vlm_outputs = self.vlm(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            **tokenized_data,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()
        rope_deltas = self.vlm.model.rope_deltas

        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = torch.full((B,), prefill_seq_len, device=device, dtype=torch.long)

        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
        delta = rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # No padding between prompt cache and action block: full attention mask.
        attention_mask = torch.zeros(
            (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # Build flow-matching target: x_1 = GT action, x_0 ~ N(0, I).
        x_1 = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz[:, 0],
            traj_history_rot=ego_history_rot[:, 0],
            traj_future_xyz=ego_future_xyz[:, 0],
            traj_future_rot=ego_future_rot[:, 0],
        )  # (B, n_diffusion_tokens, 2)
        x_1 = x_1.to(device=device, dtype=torch.float32)

        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, 1, 1, device=device, dtype=x_1.dtype)
        x_t = (1.0 - t) * x_0 + t * x_1
        v_target = x_1 - x_0

        # Cast to action-module dtype to match action_in_proj / expert weights.
        proj_dtype = next(self.action_in_proj.parameters()).dtype
        x_t_cast = x_t.to(dtype=proj_dtype)
        t_cast = t.to(dtype=proj_dtype)

        future_token_embeds = self.action_in_proj(x_t_cast, t_cast)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(B, n_diffusion_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        v_pred = self.action_out_proj(last_hidden).view(
            B, *self.action_space.get_action_space_dims()
        )

        return {"v_pred": v_pred.to(torch.float32), "v_target": v_target}

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) run autoregressive generation for the VLM
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # find <traj_future_start> token position for each sequence, use last token if not found
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        # [b_star], True if sequence has <traj_future_start>
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"No <traj_future_start> token found in the generated sequences for sequence {i}"
                )
        # [b_star], first occurrence position
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        # note that vlm_outputs.sequences already include the input_ids,
        # so no need to add the input_ids length
        offset = valid_token_pos_id + 1

        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # modify the attention_masks to remove padding tokens
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
