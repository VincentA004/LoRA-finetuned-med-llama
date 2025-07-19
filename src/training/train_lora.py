
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer, setup_chat_format

# Import settings from the centralized configuration file
import src.config as config

def main():
    # Load the tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, trust_remote_code=True)
    
    # Configure quantization with BitsAndBytes
    bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)

    # Load the base model with the specified quantization config
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Set up chat formatting and prepare the model for k-bit training
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Print a summary of the trainable parameters
    model.print_trainable_parameters()
    
    # Load and prepare the dataset
    ds = load_dataset(config.DATASET_NAME, split="train")
    ds = ds.shuffle(seed=42).select(range(config.NUM_SAMPLES))
    
    # Helper function to format the dataset examples into a chat template
    def format_example(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]}
        ]
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example
    
    ds = ds.map(format_example, num_proc=4)
    ds = ds.train_test_split(test_size=0.1)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.LORA_ADAPTER_PATH,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        **config.TRAINING_ARGS,
    )
    
    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        args=training_args,
        peft_config=peft_config,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )
    
    # Start the training process
    print("Starting model training...")
    trainer.train()
    
    # Save the trained model adapter and tokenizer
    print(f"Saving LoRA adapter to {config.LORA_ADAPTER_PATH}...")
    trainer.model.save_pretrained(config.LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(config.LORA_ADAPTER_PATH)
    
    print(f"Training complete. Model adapter saved to {config.LORA_ADAPTER_PATH}")

if __name__ == "__main__":
    main()