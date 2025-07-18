import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def load_model(model_path: str) -> ort.InferenceSession:
    print(f"[INFO] Loading ONNX model from: {model_path}")
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def prepare_inputs(tokenizer, prompt: str, max_length: int):
    enc = tokenizer(prompt, return_tensors="np", padding=True, truncation=True, max_length=max_length)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }, enc["input_ids"]


def run_inference(session, inputs):
    start = time.perf_counter()
    outputs = session.run(None, inputs)
    end = time.perf_counter()
    return outputs, end - start


def decode_output(tokenizer, output_logits, input_ids):
    generated_ids = np.argmax(output_logits[0], axis=-1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a quantized ONNX model.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum input sequence length.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")  # Adjust as needed

    session = load_model(str(model_path))
    inputs, input_ids = prepare_inputs(tokenizer, args.prompt, args.max_length)
    outputs, latency = run_inference(session, inputs)

    output_text = decode_output(tokenizer, outputs, input_ids)

    print("\n[OUTPUT]")
    print(output_text.strip())
    print(f"\n[METRICS] Inference latency: {latency:.4f} seconds")


if __name__ == "__main__":
    main()
