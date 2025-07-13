#!/usr/bin/env python3

"""
load_vllm_bamba_skip_conv.py

Example script to:
- Load Bamba-9B-v2 from IBM via vLLM with skip convolution
- Print model config details
- Wrap with DeepEval adapter
- Run MMLU benchmark

Usage:
    python load_vllm_bamba_skip_conv.py
"""

# -------------------------
# Imports
# -------------------------

from vllm import LLM, SamplingParams
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM

# Import the skip convolution functions
from vllm.model_executor.layers.mamba.ops.casual_conv1d_skip import (
    causal_conv1d_fn_skip_conv, causal_conv1d_update_skip_conv
)

# Import the original causal convolution functions to monkey patch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update
)

# Import the MambaMixer2 to patch it
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2


# -------------------------
# Helper class to match DeepEval's expected output
# -------------------------
class SimpleLLMResult:
    def __init__(self, answer: str):
        self.answer = answer


# -------------------------
# Adapter class to wrap vLLM's LLM for DeepEval
# -------------------------
class BambaModelAdapter(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    def load_model(self):
        # No-op for vLLM
        pass

    def get_model_name(self) -> str:
        return "ibm-ai-platform/Bamba-9B-v2-skip-conv"

    def generate(self, prompt: str, **kwargs):
        outputs = self.llm.generate([prompt], self.sampling_params)
        prediction = outputs[0].outputs[0].text.strip()
        return SimpleLLMResult(answer=prediction)

    async def a_generate(self, prompt: str, **kwargs):
        return self.generate(prompt)


# -------------------------
# Monkey patch function to replace causal convolution with skip convolution
# -------------------------
def patch_mamba_with_skip_conv(skip_initial: int = 1):
    """
    Monkey patch the MambaMixer2 to use skip convolution instead of regular causal convolution.
    
    Args:
        skip_initial: Number of initial tokens to skip in convolution (default: 1)
    """
    
    # Store original functions
    original_causal_conv1d_fn = causal_conv1d_fn
    original_causal_conv1d_update = causal_conv1d_update
    
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
    print("\n=== Loading IBM Bamba-9B-v2 via vLLM with Skip Convolution ===")

    # Patch the model with skip convolution before loading
    patch_mamba_with_skip_conv(skip_initial=1)

    llm = LLM(
        model="ibm-ai-platform/Bamba-9B-v2",
        max_model_len=100000,
        gpu_memory_utilization=0.95
    )

    print("\n=== Model Configuration ===")
    model_config = llm.llm_engine.get_model_config()

    print("\nAvailable model_config attributes:")
    print(dir(model_config))

    print("\nDetails:")
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

    if hasattr(model_config, "attention_chunk_size"):
        print("Attention chunk size:", model_config.attention_chunk_size)

    # -------------------------
    # Wrap model with Adapter
    # -------------------------
    print("\n=== Wrapping Model with DeepEval Adapter ===")
    adapter = BambaModelAdapter(llm)

    # -------------------------
    # Define MMLU Benchmark
    # -------------------------
    print("\n=== Defining MMLU Benchmark ===")
    benchmark = MMLU(
        tasks=[
            MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
            MMLUTask.ASTRONOMY
        ],
        n_shots=3
    )

    # -------------------------
    # Run Evaluation
    # -------------------------
    print("\n=== Running Evaluation ===")
    benchmark.evaluate(model=adapter)

    print("\n=== MMLU Results ===")
    print(f"MMLU Accuracy: {benchmark.overall_score:.2%}")


# -------------------------
# Script entry point
# -------------------------
if __name__ == "__main__":
    main() 