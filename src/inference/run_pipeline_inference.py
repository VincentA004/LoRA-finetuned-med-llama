import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from trl import setup_chat_format

def build_prompt(tokenizer, message):
    messages = [{"role": "user", "content": message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser(description="Chat-style inference with LoRA fine-tuned LLaMA model using HF pipeline.")
    parser.add_argument("--model_path", type=str, default="models/merged", help="Path to the merged model directory")
    parser.add_argument("--prompt", type=str, default="Hello doctor, I have bad acne. How do I get rid of it?", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--use_fp16", action="store_true", help="Enable float16 precision")
    args = parser.parse_args()

    logging.set_verbosity_error()

    # Device config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32

    # Load tokenizer and model
    print(f"🔄 Loading tokenizer and model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype
    )
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Build prompt and run inference
    prompt = build_prompt(tokenizer, args.prompt)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    output = pipe(
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Extract clean response
    response = output[0]["generated_text"].split("assistant")[-1].strip()
    print(f"\n🧠 Model response:\n{response}")

if __name__ == "__main__":
    main()
