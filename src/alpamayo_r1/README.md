<div align="center">

# Current Progress

</div>

### 0. Overview
Please follow https://github.com/cehongwang/alpamayo/blob/main/README.md to install the environment. You may switch the PyTorch version to test the script. 2.8.0 may have worse accuracy than 2.9.0 based on the test. Also tried the torch nightly branch but saw some errors in the Huggingface library.

After installing the env, you can run `python compile_torchtrt.py` to start the compilation.

Options:
At the top of the script, you can find the script settings that help you test different configurations. 

- `debug_mode`: whether to initialize one-layer model to accelerate the debug process. Compiling the whole model can take more than 5 mins. However, this can only be used to test the **TRT compilation process**, not the whole model. Running the pipeline with only a one-layer model can give errors.

- `use_cache`: whether to enable KV cache

- `cache_implementation`: dynamic/static implementation of the KV cache. Currently, Alpamayo only supports dynamic KV cache implementation, but that is not torch.export compatible. We might want to switch to the static KV cache implementation for Torch-TensorRT.

- `plot`: whether to plot error distribution. The plot will be saved to the current working directory.
To compile each segment of the model, you can uncomment the code after the Debugger context manager.

### 1. Compile Vision Model
By uncommenting the `compile_qwen3vl_visual`, you can initialize the vision part compilation. 

- Model compilation: ✓
- Accuracy: ✖ Max diff: 0.65625, Mean diff: 0.0025634765625 (Torch 2.8); 0.01 (Torch 2.9)

### 2. Compile Language Model
By uncommenting the `compile_qwen3vl_language_model`, you can initialize the language part compilation. 

- Model compilation: ✓ 
- KV cache support: ✖
- Accuracy: ✖ Max diff: 61.0, Mean diff: 0.06201171875. Tried strong typing and weaktyping and fp_32_acc. Doesn't make a difference. (Torch 2.8.0) 

### 3. Compile the Diffusion Model
By uncommenting the `compile_diffusion`, you can initialize the diffusion part compilation. 

- Model compilation: ✓ in Torch 2.9; fails on 2.8.0 
- Accuracy: ✖ Max diff: ~1.0
