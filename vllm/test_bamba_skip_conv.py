#!/usr/bin/env python3

"""
test_bamba_skip_conv.py

Simple test script to:
- Load Bamba-9B-v2 from IBM via vLLM with skip convolution
- Test basic inference with skip convolution
- Compare with regular convolution (if needed)

Usage:
    python test_bamba_skip_conv.py
"""

# -------------------------
# Imports
# -------------------------

from vllm import LLM, SamplingParams

# Import the skip convolution functions
from vllm.model_executor.layers.mamba.ops.casual_conv1d_skip import (
    causal_conv1d_fn_skip_conv, causal_conv1d_update_skip_conv
)

# Import the original causal convolution functions to monkey patch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update
)


# -------------------------
# Monkey patch function to replace causal convolution with skip convolution
# -------------------------
def patch_mamba_with_skip_conv(skip_initial: int = 1):
    """
    Monkey patch the MambaMixer2 to use skip convolution instead of regular causal convolution.
    
    Args:
        skip_initial: Number of initial tokens to skip in convolution (default: 1)
    """
    
    # Create wrapper functions that use skip convolution
    def patched_causal_conv1d_fn(*args, **kwargs):
        return causal_conv1d_fn_skip_conv(*args, skip_initial=skip_initial, **kwargs)
    
    def patched_causal_conv1d_update(*args, **kwargs):
        return causal_conv1d_update_skip_conv(*args, skip_initial=skip_initial, **kwargs)
    
    # Monkey patch the functions in the causal_conv1d module
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as causal_conv1d_module
    causal_conv1d_module.causal_conv1d_fn = patched_causal_conv1d_fn
    causal_conv1d_module.causal_conv1d_update = patched_causal_conv1d_update
    
    print(f"✅ Successfully patched Mamba with skip convolution (skip_initial={skip_initial})")


# -------------------------
# Main function
# -------------------------
def main():
    print("\n=== Testing IBM Bamba-9B-v2 with Skip Convolution ===")

    # Patch the model with skip convolution before loading
    patch_mamba_with_skip_conv(skip_initial=1)

    print("\n=== Loading Model ===")
    llm = LLM(
        model="ibm-ai-platform/Bamba-9B-v2",
        max_model_len=100000,
        gpu_memory_utilization=0.95
    )

    print("\n=== Model Configuration ===")
    model_config = llm.llm_engine.get_model_config()

    print("Model:", model_config.model)
    print("Max model length:", model_config.max_model_len)

    if hasattr(model_config, "get_hidden_size"):
        print("Hidden size:", model_config.get_hidden_size())

    parallel_config = llm.llm_engine.parallel_config
    print("Number of layers:", model_config.get_num_layers(parallel_config))

    if hasattr(model_config, "get_num_attention_heads"):
        print("Number of attention heads:", model_config.get_num_attention_heads(parallel_config))

    if hasattr(model_config, "get_vocab_size"):
        print("Vocab size:", model_config.get_vocab_size())

    # -------------------------
    # Test basic inference
    # -------------------------
    print("\n=== Testing Basic Inference ===")
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers."
    ]
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50,
        top_p=0.9
    )
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        try:
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during inference: {e}")
    
    print("\n=== Skip Convolution Test Complete ===")
    print("The model is now using skip convolution instead of regular causal convolution.")
    print("You can compare the outputs with the original model to see the differences.")


# -------------------------
# Script entry point
# -------------------------
if __name__ == "__main__":
    main() 