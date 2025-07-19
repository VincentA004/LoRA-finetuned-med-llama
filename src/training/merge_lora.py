
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import setup_chat_format

# Import settings from the centralized configuration file
import src.config as config

def main():
    """
    Merges the fine-tuned LoRA adapter with the base model and saves the
    resulting model and tokenizer to a new directory.
    """
    print(f"Loading base model: {config.BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Attaching LoRA adapter from: {config.LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, config.LORA_ADAPTER_PATH)
    
    print("Merging adapter weights into the base model...")
    model = model.merge_and_unload()
    
    print("Loading tokenizer and setting up chat format...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    print(f"Saving merged model to: {config.MERGED_MODEL_PATH}")
    model.save_pretrained(config.MERGED_MODEL_PATH)
    tokenizer.save_pretrained(config.MERGED_MODEL_PATH)
    
    print(f"Merge complete. Merged model and tokenizer saved to {config.MERGED_MODEL_PATH}")

if __name__ == "__main__":
    main()