import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

# === CONFIG ===
ONNX_MODEL_PATH = "models/onnx/model.onnx"

def quantize(quant_type: str):
    quant_type_map = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8
    }

    if quant_type not in quant_type_map:
        print("Error: Invalid quant_type. Use 'int8' or 'uint8'.")
        return

    output_path = f"models/onnx/model.{quant_type}.onnx"

    print(f"Quantizing ONNX model to {quant_type.upper()}...")
    quantize_dynamic(
        model_input=ONNX_MODEL_PATH,
        model_output=output_path,
        weight_type=quant_type_map[quant_type]
    )
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quantize_onnx.py <quant_type>")
        print("Example: python quantize_onnx.py int8")
    else:
        quantize(sys.argv[1])
