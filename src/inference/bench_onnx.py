import time
import argparse
import numpy as np
import psutil
import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from tabulate import tabulate

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_model(model_path: str):
    model = ORTModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def benchmark(model, tokenizer, prompt: str, max_new_tokens: int = 64, iterations: int = 5, print_stats: bool = True):
    tokens_per_second = []
    latencies = []
    memory_usages = []

    inputs = tokenizer(prompt, return_tensors="pt")

    for i in range(iterations):
        mem_before = get_memory_mb()
        start = time.perf_counter()

        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )

        end = time.perf_counter()
        mem_after = get_memory_mb()

        total_time = end - start
        latency = total_time / max_new_tokens
        tokens_per_sec = max_new_tokens / total_time
        delta_mem = mem_after - mem_before

        latencies.append(latency * 1000)  # ms/token
        tokens_per_second.append(tokens_per_sec)
        memory_usages.append(mem_after)

        if print_stats:
            print(f"[{i+1}] {tokens_per_sec:.2f} tok/sec | {latency*1000:.2f} ms/token | +{delta_mem:.2f} MB")

    if print_stats:
        summary = [
            ["Avg tokens/sec", f"{np.mean(tokens_per_second):.2f}"],
            ["Avg latency (ms/token)", f"{np.mean(latencies):.2f}"],
            ["Avg memory usage (MB)", f"{np.mean(memory_usages):.2f}"],
            ["Peak memory usage (MB)", f"{np.max(memory_usages):.2f}"]
        ]
        print("\nBenchmark Summary")
        print(tabulate(summary, headers=["Metric", "Value"], tablefmt="github"))

    return {
        "tokens_per_second": tokens_per_second,
        "latencies_ms": latencies,
        "memory_usages_mb": memory_usages,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX quantized model")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model directory")
    parser.add_argument("--prompt", type=str, default="What are the symptoms of Type 2 diabetes?", help="Prompt for benchmarking")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--iters", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--no-print", action="store_true", help="Disable stats printout (for notebook use)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    benchmark(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        iterations=args.iters,
        print_stats=not args.no_print,
    )

if __name__ == "__main__":
    main()
