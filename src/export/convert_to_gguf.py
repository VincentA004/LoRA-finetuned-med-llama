import os
import subprocess

# === CONFIG ===
MERGED_MODEL_PATH = "models/merged"
GGUF_OUTPUT_PATH = "models/gguf/llama3-med.gguf"
CONVERT_SCRIPT = "llama.cpp/scripts/convert.py"  # Update this path if needed
PYTHON = "python"  # Change to "python3" if required on your system

def convert_to_gguf():
    os.makedirs(os.path.dirname(GGUF_OUTPUT_PATH), exist_ok=True)

    command = [
        PYTHON,
        CONVERT_SCRIPT,
        MERGED_MODEL_PATH,
        "--outfile", GGUF_OUTPUT_PATH,
        "--outtype", "f16"  # You can change to q4_0 for quantized output
    ]

    print("Running GGUF conversion...")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"GGUF model exported successfully to {GGUF_OUTPUT_PATH}")
    else:
        print("GGUF conversion failed:")
        print(result.stderr)

if __name__ == "__main__":
    convert_to_gguf()
