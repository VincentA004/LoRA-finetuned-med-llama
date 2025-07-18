#!/bin/bash

# Benchmark GGUF model using llama.cpp's benchmark tool

MODEL_PATH=$1
N_TOKENS=${2:-128}
N_THREADS=${3:-8}
N_BATCH=${4:-32}
REPEAT=${5:-5}

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <path_to_model.gguf> [n_tokens] [n_threads] [n_batch] [repeat]"
  exit 1
fi

echo "---------------------------------------"
echo "Benchmarking GGUF model: $MODEL_PATH"
echo "Tokens: $N_TOKENS | Threads: $N_THREADS | Batch: $N_BATCH | Repeat: $REPEAT"
echo "---------------------------------------"

./llama.cpp/benchmark \
  -m "$MODEL_PATH" \
  -t "$N_THREADS" \
  -n "$N_TOKENS" \
  -b "$N_BATCH" \
  --repeat "$REPEAT"
