
import torch

# ==============================================================================
# Model and Data Configuration
# ==============================================================================
# Base model identifier from the Hugging Face Hub. This is the foundational
# model that will be fine-tuned.
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Dataset identifier from the Hugging Face Hub used for fine-tuning.
DATASET_NAME = "FiscaAI/synth-ehr-icd10cm-prompt"

# Specifies the number of samples to select from the dataset for training.
# This is useful for running experiments on a smaller subset of the data.
NUM_SAMPLES = 3000


# ==============================================================================
# File and Directory Paths
# ==============================================================================
# Root directory for all generated artifacts, including models and logs.
OUTPUT_DIR_BASE = "models"

# Directory to save the trained LoRA adapter weights.
LORA_ADAPTER_PATH = f"{OUTPUT_DIR_BASE}/llama3-lora-icd"

# Directory to save the final, merged model after applying the LoRA adapter.
MERGED_MODEL_PATH = f"{OUTPUT_DIR_BASE}/merged"

# Directory for storing the ONNX-exported version of the model.
ONNX_EXPORT_PATH = f"{OUTPUT_DIR_BASE}/onnx"

# Full path to the ONNX model file.
ONNX_MODEL_FILE = f"{ONNX_EXPORT_PATH}/model.onnx"

# Directory for storing GGUF-formatted models.
GGUF_OUTPUT_DIR = f"{OUTPUT_DIR_BASE}/gguf"

# Full path for the unquantized (FP16) GGUF model file.
GGUF_FP16_FILE = f"{GGUF_OUTPUT_DIR}/llama3-med.gguf"

# Local path to the cloned llama.cpp repository, required for GGUF tooling.
# This path must be correctly configured for GGUF conversion and quantization.
LLAMA_CPP_PATH = "llama.cpp"


# ==============================================================================
# Training Hyperparameters
# ==============================================================================
# Total number of complete passes through the training dataset.
NUM_TRAIN_EPOCHS = 1

# Maximum token length for model inputs during training and inference.
MAX_SEQ_LENGTH = 512

# LoRA (Low-Rank Adaptation) configuration parameters.
LORA_R = 16  # The rank of the update matrices.
LORA_ALPHA = 32  # The scaling factor for the LoRA weights.
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj"
]

# BitsAndBytes configuration for 4-bit quantization during training (QLoRA).
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",  # NormalFloat4 for quantization.
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}

# Hugging Face `TrainingArguments` for the SFTTrainer.
TRAINING_ARGS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "epoch",
    "fp16": True,  # Use 16-bit precision for training.
    "optim": "paged_adamw_32bit", # Paged AdamW optimizer for memory efficiency.
    "report_to": "none",  # Disable integrations like W&B/TensorBoard.
}


# ==============================================================================
# Inference and Deployment Settings
# ==============================================================================
# A default prompt used for quick inference tests and benchmarks.
DEFAULT_PROMPT = "Patient presents with shortness of breath and chest tightness. What are the likely ICD-10 codes?"

# Network host for the API server. '127.0.0.1' for local access.
SERVER_HOST = "127.0.0.1"

# Network port for the API server.
SERVER_PORT = 8000


# ==============================================================================
# Publishing Configuration
# ==============================================================================
# The Hugging Face Hub organization or username to publish the model to.
# **Action Required**: This value must be updated to your personal or org username.
HF_ORG_NAME = "your-hf-username"

# The desired repository name for the model on the Hugging Face Hub.
HF_REPO_NAME = "lora-finetuned-med-llama-3"