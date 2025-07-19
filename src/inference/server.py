
import argparse
import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM # Import the ONNX model class

import src.config as config

# --- Globals for Model and Arguments ---
model_pipeline = None
ARGS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle model loading at startup based on CLI args.
    """
    global model_pipeline
    print(f"--- Loading model (format: {ARGS.format}) ---")

    model_path = ""
    model_class = None

    # Determine which model class and path to use
    if ARGS.format == "safetensor":
        model_path = config.MERGED_MODEL_PATH
        model_class = AutoModelForCausalLM
    elif ARGS.format == "onnx":
        model_path = config.ONNX_EXPORT_PATH
        model_class = ORTModelForCausalLM
    else:
        raise ValueError(f"Unsupported model format: {ARGS.format}")

    # Load the appropriate model and tokenizer
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create the text-generation pipeline
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    print(f"--- Model '{ARGS.format}' loaded successfully. ---")
    yield
    # Clean up resources on shutdown
    global model_pipeline
    model_pipeline = None
    print("--- Model unloaded. ---")


app = FastAPI(lifespan=lifespan)


# --- Pydantic Models for Request and Response ---
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

class InferenceResponse(BaseModel):
    text: str


def run_blocking_inference(request: InferenceRequest) -> dict:
    """
    Runs the model pipeline in a separate thread. This function works for
    both SafeTensor and ONNX models via the pipeline abstraction.
    """
    if model_pipeline is None:
        raise RuntimeError("Model pipeline is not initialized.")
        
    messages = [{"role": "user", "content": request.prompt}]
    formatted_prompt = model_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    output = model_pipeline(
        formatted_prompt,
        max_new_tokens=request.max_new_tokens,
        do_sample=True,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p
    )
    
    generated_text = output[0]['generated_text']
    assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
    
    return {"text": assistant_response.strip()}


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_blocking_inference, request)
    return result


@app.get("/health")
async def health_check():
    return {"status": "ok", "format": ARGS.format if ARGS else "unknown"}


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the FastAPI inference server.")
    parser.add_argument(
        "--format",
        type=str,
        default="safetensor",
        choices=["safetensor", "onnx"],
        help="The model format to serve."
    )
    ARGS = parser.parse_args()

    # Run the server using uvicorn
    uvicorn.run(
        "server:app", # Use "module:app" string format for reloader
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=True # Add reloader for development
    )