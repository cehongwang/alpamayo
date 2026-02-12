<div align="center">

# Current Progress

</div>

### 0. Overview
You can run `python compile_torchtrt.py` to run the compilation.

Options:
At the top of the script, you can find the script settings that helps you test different config. 

- `debug_mode` whether to initialize one-layer model to accelerate the debug process. Compiling the whole model can take more than 5 mins. However, this can only be used to test the **TRT compilation process**, not the whole model. Running the pipeline with only one-layer model can give errors.

- `use_cache` whether to enable KV cache

- `cache_implementation` dynamic/static implementation of the KV cache. Currently Alpamayo only supports dynamic KV cache implementation, but that is not torch.export compatible. We might want to switch to static implementation for Torch-TensorRT.

- `plot` whether plot error distribution. The plot will be saved to the current working directory.
For compile each segment of the model, you can uncomment the code after the Debugger context manager.

### 1. Compile Vision Model
By uncomment the `compile_qwen3vl_visual`, you can initialize the vision part compilation. 

- Model compilation: ✓
- Accuracy: ✖ Max diff: 0.65625, Mean diff: 0.0025634765625

### 2. Compile Language Model
By uncomment the `compile_qwen3vl_language_model`, you can initialize the language part compilation. 

- Model compilation: ✓ 
- KV cache support: ✖
- Accuracy: ✖ Max diff: 61.0, Mean diff: 0.06201171875. Tried strong typing and weaktyping and fp_32_acc. Doesn't make a difference.

### 3. Compile the Diffusion Model
By uncomment the `compile_diffusion`, you can initialize the diffusion part compilation. 

- Model compilation: ✓ Fails on 2.8.0 
- Accuracy: ✖ Max diff: ~1.0
