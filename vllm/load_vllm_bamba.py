#!/usr/bin/env python3

"""
load_vllm_bamba.py

Example script to:
- Load Bamba-9B-v2 from IBM via vLLM
- Print model config details
- Wrap with DeepEval adapter
- Run MMLU benchmark

Usage:
    python load_vllm_bamba.py
"""

# -------------------------
# Imports
# -------------------------

from vllm import LLM, SamplingParams
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM


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
        return "ibm-ai-platform/Bamba-9B-v2"

    def generate(self, prompt: str, **kwargs):
        outputs = self.llm.generate([prompt], self.sampling_params)
        prediction = outputs[0].outputs[0].text.strip()
        return SimpleLLMResult(answer=prediction)

    async def a_generate(self, prompt: str, **kwargs):
        return self.generate(prompt)


# -------------------------
# Main function
# -------------------------
def main():
    print("\n=== Loading IBM Bamba-9B-v2 via vLLM ===")

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
