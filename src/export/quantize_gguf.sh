#!/bin/bash

# === CONFIG ===
LLAMA_CPP_PATH=llama.cpp
GGUF_INPUT=models/gguf/llama3-med.gguf
GGUF_OUTPUT_DIR=models/gguf

# Available quant types: Q4_0, Q4_K_M, Q5_0, Q5_K_S, Q8_0, F16
QUANT_TYPE=$1

if [ -z "$QUANT_TYPE" ]; then
  echo "Usage: ./quantize_gguf.sh <QUANT_TYPE>"
  echo "Example: ./quantize_gguf.sh Q4_0"
  exit 1
fi

${LLAMA_CPP_PATH}/quantize \
  ${GGUF_INPUT} \
  ${GGUF_OUTPUT_DIR}/llama3-med.${QUANT_TYPE}.gguf \
  ${QUANT_TYPE}

echo "Quantized GGUF model saved to ${GGUF_OUTPUT_DIR}/llama3-med.${QUANT_TYPE}.gguf"
