# Static-Cache + Aliased-I/O LM Implementation Plan

> **For agentic workers:** implement task-by-task; steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the custom `PrefixKVCache` LM-decode machinery with HuggingFace-native
`StaticCache` wrapped for `torch.export`, and compile it to a Torch-TensorRT engine whose
KV buffers become aliased in-place I/O — keeping the native `Qwen3VLTextModel` path and
changing as little around it as possible.

**Architecture:** Start from the native `Qwen3VLTextModel`. Wrap it in a thin static-cache
module modeled on HF's `TorchExportableModuleWithStaticCache` (reuse `transformers`'
`StaticCache`), adding only the Qwen3-VL plumbing the stock wrapper omits (M-RoPE
`position_ids`, DeepStack). Export -> TRT compile with `decompose_attention=False`,
relying on the fork's buffer-lifting to emit `IKVCacheUpdateLayer` aliased writes. Drive
decode with a manual single-token loop; hand the filled cache to the diffusion expert by
reading the static buffers.

**Tech Stack:** transformers (StaticCache, executorch integration), Torch-TensorRT dynamo
(`/home/TensorRT`), Qwen3-VL.

**Guiding constraint:** least change from HF native, best decode perf.

## Scope note (discovered during implementation)
`PrefixKVCache` is also consumed by `diffusion.py` (read-only prefix feeder for the expert
during diffusion export) and as a plugin fallback. Therefore we remove only the LM-decode
machinery (`lm_with_cache.py`, `extract_stacked_kv_from_cache`, `stack_prefix_kv_from_cache`)
and KEEP `PrefixKVCache` for the diffusion feeder. The LM decode no longer uses it.

---

## Task 1: Native static-cache module on the bare text model (eager)
**Files:** Create `src/alpamayo_r1/trt/static_cache_lm.py`
- [ ] `Qwen3VLStaticCacheLM` nn.Module: builds `StaticCache` + `early_initialization`,
  registers `key_cache_i/value_cache_i/cumulative_length_i` buffers; forward
  `(inputs_embeds, cache_position, position_ids, visual_pos_masks, deepstack_visual_embeds)`
  resets `cumulative_length` to `cache_position[0:1]` and calls the text model, returns
  `last_hidden_state`.
- [ ] `reset_static_cache(module)` zeros cache + cumulative_length buffers.
- [ ] Eager smoke test vs DynamicCache reference (GPU).

## Task 2: Export + TRT compile with native attention + aliased KV
**Files:** modify `static_cache_lm.py`
- [ ] `export_static_cache_lm`: GQA-off during export, dynamic batch, static seq=1.
- [ ] `compile_static_cache_lm_trt`: `decompose_attention=False`, fp16, C++ runtime;
  assert non-empty `aliased_io` (== 2*num_layers bindings).

## Task 3: Manual decode loop + wire into compile_trt
**Files:** modify `static_cache_lm.py`, `compile_trt.py`
- [ ] `compile_vlm_lm_trt_static_cache(model, ...)` entry (replaces
  `compile_vlm_lm_trt_with_cache`).
- [ ] `run_vlm_rollout_static(...)` prefill-once + single-token decode using the TRT module,
  sampling via existing `sample_token`, EOS handling.
- [ ] `compile_language_trt` imports the new compiler; `run_inference_trt` uses the static
  rollout.

## Task 4: Diffusion handoff from static buffers
**Files:** modify `prefix_cache.py`, `compile_trt.py`
- [ ] `stack_static_kv(cache, seq_len)` reads StaticCache layer buffers -> `[L,B,H,S,D]`.
- [ ] `run_inference_trt` feeds the diffusion `step_fn` from `stack_static_kv`.

## Task 5: Delete LM-decode dead code
**Files:** delete `lm_with_cache.py`; trim `prefix_cache.py`
- [ ] Remove `extract_stacked_kv_from_cache`, `stack_prefix_kv_from_cache`.
- [ ] Keep `PrefixKVCache` + `maybe_to` (diffusion feeder).
- [ ] Delete `lm_with_cache.py`; update imports.

## Risks
- M-RoPE under export; DeepStack fixed-shape input; prefill vs decode shapes; max_cache_len
  sizing; fp16 drift (validate via minADE).
