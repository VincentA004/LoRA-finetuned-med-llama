
import argparse
import os
import subprocess
from pathlib import Path

from optimum.exporters.onnx import main_export
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.config as config


def export_to_onnx():
    """Exports the merged model to the ONNX format."""
    print("--- Starting ONNX Export ---")
    output_path = Path(config.ONNX_EXPORT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    # Note: ONNX export for Llama3 requires the development version of transformers
    # and a specific version of optimum. Ensure your environment is compatible.
    main_export(
        model_name_or_path=config.MERGED_MODEL_PATH,
        output=output_path,
        task="text-generation",
        opset=13,  # Llama3 compatibility
        do_validation=False
    )
    print(f"✅ ONNX model saved to {config.ONNX_EXPORT_PATH}")


def export_to_gguf():
    """Converts the merged model to an FP16 GGUF file using llama.cpp."""
    print("--- Starting GGUF Export ---")
    gguf_dir = Path(config.GGUF_OUTPUT_DIR)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    
    convert_script = Path(config.LLAMA_CPP_PATH) / "convert.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert.py script not found at {convert_script}. "
            "Please ensure LLAMA_CPP_PATH is configured correctly in src/config.py."
        )

    command = [
        "python", str(convert_script), config.MERGED_MODEL_PATH,
        "--outfile", config.GGUF_FP16_FILE,
        "--outtype", "f16"  # Export to FP16 first, then quantize
    ]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ GGUF (FP16) model successfully exported to {config.GGUF_FP16_FILE}")
    else:
        print("❌ GGUF conversion failed.")
        print(result.stderr)
        raise RuntimeError("GGUF conversion failed.")


def quantize_gguf(quant_type: str):
    """Quantizes an FP16 GGUF model to a specified format using llama.cpp."""
    print(f"--- Starting GGUF Quantization ({quant_type}) ---")
    quantize_script = Path(config.LLAMA_CPP_PATH) / "quantize"
    if not quantize_script.exists():
        raise FileNotFoundError(
            f"llama.cpp quantize executable not found at {quantize_script}. "
            "Please ensure llama.cpp is compiled and LLAMA_CPP_PATH is correct."
        )
    
    output_file = Path(config.GGUF_OUTPUT_DIR) / f"llama3-med.{quant_type}.gguf"

    command = [
        str(quantize_script),
        config.GGUF_FP16_FILE,
        str(output_file),
        quant_type
    ]

    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ GGUF model successfully quantized to {output_file}")
    else:
        print("❌ GGUF quantization failed.")
        print(result.stderr)
        raise RuntimeError("GGUF quantization failed.")


def main():
    parser = argparse.ArgumentParser(description="Export and quantize the fine-tuned model.")
    parser.add_argument("--format", type=str, required=True, choices=["onnx", "gguf"], help="The format to export to.")
    parser.add_argument("--quantize", type=str, help="GGUF quantization type (e.g., Q4_K_M, Q8_0).")
    
    args = parser.parse_args()

    if args.format == "onnx":
        export_to_onnx()
    elif args.format == "gguf":
        export_to_gguf()
        if args.quantize:
            quantize_gguf(args.quantize)

if __name__ == "__main__":
    main()