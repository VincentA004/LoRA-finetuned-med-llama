import os
import torch
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


# === CONFIG ===
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "FiscaAI/synth-ehr-icd10cm-prompt"
OUTPUT_DIR = "models/llama3-lora-icd"
NUM_SAMPLES = 3000
EPOCHS = 1
MAX_SEQ_LEN = 512

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = load_dataset(DATASET_NAME, split="train")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))

    def format_example(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]}
        ]
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    ds = ds.map(format_example, num_proc=4)
    ds = ds.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=EPOCHS,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        args=training_args,
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f" Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
