import os
import argparse
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def push_lora_adapter(lora_path, repo_id, private):
    print(f"Pushing LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(lora_path)
    model.push_to_hub(repo_id, private=private)
    print("✅ LoRA adapter pushed.")

def push_merged_model(merged_path, repo_id, private):
    print(f"Pushing merged model from: {merged_path}")
    model = AutoModelForCausalLM.from_pretrained(merged_path)
    model.push_to_hub(repo_id, private=private)
    print("✅ Merged model pushed.")

def push_tokenizer(tokenizer_path, repo_id, private):
    print(f"Pushing tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.push_to_hub(repo_id, private=private)
    print("✅ Tokenizer pushed.")

def push_gguf_file(gguf_path, repo_id, hf_token):
    if gguf_path and os.path.exists(gguf_path):
        print(f"Pushing GGUF file from: {gguf_path}")
        api = HfApi()
        filename = os.path.basename(gguf_path)
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print("✅ GGUF file uploaded.")
    else:
        print("⚠️ No GGUF file provided or path does not exist. Skipping.")

def get_hf_token(cli_token: str = None) -> str:
    return (
        cli_token or
        os.getenv("HF_TOKEN") or
        HfFolder.get_token()
    )

def main():
    parser = argparse.ArgumentParser(description="Push all model artifacts to Hugging Face Hub.")
    parser.add_argument("--hf-repo-name", required=True, help="Name of the Hugging Face model repo")
    parser.add_argument("--hf-org", required=True, help="Your Hugging Face organization or username")
    parser.add_argument("--lora-path", required=True, help="Path to saved LoRA adapter folder")
    parser.add_argument("--merged-path", required=True, help="Path to merged model folder")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--gguf-path", help="Path to GGUF file (optional)")
    parser.add_argument("--hf-token", help="Hugging Face token (optional)")
    parser.add_argument("--private", action="store_true", help="Upload model privately")

    args = parser.parse_args()
    hf_token = get_hf_token(args.hf_token)
    repo_id = f"{args.hf_org}/{args.hf_repo_name}"

    print(f"📤 Uploading to: https://huggingface.co/{repo_id}")
    push_lora_adapter(args.lora_path, repo_id, args.private)
    push_merged_model(args.merged_path, repo_id, args.private)
    push_tokenizer(args.tokenizer_path, repo_id, args.private)
    push_gguf_file(args.gguf_path, repo_id, hf_token)

    print("🎉 All uploads completed successfully.")

if __name__ == "__main__":
    main()
