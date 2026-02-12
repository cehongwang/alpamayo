import torch
import torch.nn.functional as F
from torch import nn
import torch_tensorrt
import numpy as np
import argparse
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from contextlib import nullcontext
#============================================================== 
# Script settings
#==============================================================
debug_mode = False
use_cache = False
cache_implementation = "dynamic"
plot = True
#==============================================================

torch.cuda.manual_seed_all(42)

torch._dynamo.config.capture_scalar_outputs = True

# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda").eval()
processor = helper.get_processor(model.tokenizer)

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

model_inputs = helper.to_device(model_inputs, "cuda")
print("model_inputs:\n", model_inputs)

###################### submodels ######################
# model.vlm: Qwen3VLForConditionalGeneration
# - model.vlm.model.visual: Qwen3VLVisionModel
# - model.vlm.model.language_model: Qwen3VLTextModel
# model.diffusion: FlowMatching
# - model.expert: Qwen3VLTextModel
#######################################################
model.config.attn_implementation = "sdpa"
# register_sdpa.enable_sdpa_converter("default", model.vlm.model.visual.config)

if debug_mode:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers import AutoConfig, AutoModel, StoppingCriteriaList

    config = model.vlm.config
    config.text_config.num_hidden_layers = 1
    config.vision_config.depth = 1
    model.vlm = Qwen3VLForConditionalGeneration(config)
    model.expert = AutoModel.from_config(config.text_config)

model.to(torch.bfloat16).to("cuda").eval()

with torch_tensorrt.dynamo.Debugger(log_level="debug", logging_dir="/home/profile/logging/alpamayo_r1", engine_builder_monitor=False) if debug_mode else nullcontext():

    # 1. Compile model.vlm.model.visual
    from alpamayo_r1.compile_utils import compile_qwen3vl_visual
    model.vlm.model.visual.forward = compile_qwen3vl_visual(model.vlm.model.visual, model_inputs, plot=plot).forward


    # 2. Compile model.vlm.model.language_model
    from alpamayo_r1.compile_utils import compile_qwen3vl_language_model
    model.vlm.model.language_model.forward = compile_qwen3vl_language_model(model.vlm.model.language_model, model_inputs, use_cache, cache_implementation, plot=plot).forward


    # 3. Compile model.diffusion
    # from alpamayo_r1.compile_utils import compile_diffusion, CompiledDiffusion
    # trt_diffusion = compile_diffusion(model)
    # model.diffusion.sample = CompiledDiffusion(trt_diffusion, model.diffusion.x_dims, torch.bfloat16).sample




with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
        use_cache=use_cache,
        cache_implementation=cache_implementation,
        
    )

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)
