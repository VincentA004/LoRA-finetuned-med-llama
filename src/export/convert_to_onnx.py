import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.exporters.onnx import main_export
from pathlib import Path

# === CONFIG ===
MERGED_MODEL_PATH = "models/merged"
ONNX_EXPORT_PATH = "models/onnx"

def export_to_onnx():
    print("Loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)

    print("Exporting to ONNX...")
    main_export(
        model=model,
        output=Path(ONNX_EXPORT_PATH),
        tokenizer=tokenizer,
        task="text-generation",
        opset=17,
    )

    print(f"ONNX model saved to {ONNX_EXPORT_PATH}")

if __name__ == "__main__":
    export_to_onnx()
