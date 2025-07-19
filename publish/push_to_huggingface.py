
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder

# Import settings from the centralized configuration file
import src.config as config


def get_hf_token(cli_token: str = None) -> str:
    """Retrieves the Hugging Face token from CLI, environment variables, or cache."""
    token = cli_token or os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please log in via `huggingface-cli login` or pass with --hf-token.")
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Push model artifacts to the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo-name", default=config.HF_REPO_NAME, help="Name of the Hugging Face model repo.")
    parser.add_argument("--hf-org", default=config.HF_ORG_NAME, help="Your Hugging Face organization or username.")
    parser.add_argument("--hf-token", help="Hugging Face API token (optional).")
    parser.add_argument("--private", action="store_true", help="Upload the model as a private repository.")
    
    args = parser.parse_args()
    
    hf_token = get_hf_token(args.hf_token)
    repo_id = f"{args.hf_org}/{args.repo_name}"
    
    api = HfApi(token=hf_token)
    
    print(f"↗️  Creating repository {repo_id} on Hugging Face Hub...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)
    
    # --- Upload Merged Model ---
    print(f"📤 Uploading merged model from {config.MERGED_MODEL_PATH}...")
    api.upload_folder(
        folder_path=config.MERGED_MODEL_PATH,
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: Add merged model weights and tokenizer"
    )
    
    # --- Upload GGUF Files ---
    gguf_dir = Path(config.GGUF_OUTPUT_DIR)
    if gguf_dir.exists():
        print(f"📤 Uploading GGUF models from {gguf_dir}...")
        api.upload_folder(
            folder_path=gguf_dir,
            repo_id=repo_id,
            repo_type="model",
            allow_patterns="*.gguf",
            commit_message="feat: Add GGUF quantized models"
        )

    print(f"\n🎉 Successfully uploaded artifacts to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()