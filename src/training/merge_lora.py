import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import setup_chat_format

# === CONFIG ===
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_ADAPTER_PATH = "models/llama3-lora-icd"
MERGED_MODEL_PATH = "models/merged"

def main():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )

    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.merge_and_unload()

    print("Saving merged model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model, tokenizer = setup_chat_format(model, tokenizer)

    model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print(f"Merge complete. Merged model saved to {MERGED_MODEL_PATH}")

if __name__ == "__main__":
    main()
