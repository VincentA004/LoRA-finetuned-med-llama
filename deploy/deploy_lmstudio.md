# Deploying Your GGUF Model in LM Studio

This guide explains how to run your quantized GGUF model locally using [LM Studio](https://lmstudio.ai/) — a user-friendly desktop interface for LLM inference.

LM Studio supports:
- `GGUF` models from `llama.cpp`
- Mistral, LLaMA, CodeLLaMA, and LoRA variants
- macOS, Windows, and Linux (with GPU acceleration)

## 🛠 Requirements

- Your quantized model file (e.g. `llama-3-med-Q4_K_M.gguf`)
- LM Studio installed: https://lmstudio.ai/
- Enough disk/memory (~5–8 GB recommended)

You should already have a `.gguf` file exported using `llama.cpp` tooling or Hugging Face converters.

## 📥 Step 1: Import Your GGUF Model into LM Studio

1. **Launch LM Studio** on your desktop.
2. Click **“Local Models”** from the left sidebar.
3. Then click **“Import Model”**.

You’ll be prompted to select a `.gguf` file.

4. Navigate to your exported model, for example:

    /path/to/llama-3-med-Q4_K_M.gguf


5. Optionally rename it (e.g., `MedLLaMA 3 - CPT/ICD fine-tuned`)
6. Hit **Import**.

LM Studio will now load the model and index its metadata.

## 🧪 Step 2: Run Clinical Prompts

Once imported, click your model in the **Local Models** list.

1. In the **Chat** tab, type a prompt like:

    Patient presents with shortness of breath and chest tightness. What are the likely ICD-10 codes?


2. Hit **Enter** or click **Send**.

You should see a fast response (~10–50 tokens/sec depending on quant level).

> ⚠️ Tip: If the model stalls or responds strangely, try adjusting:
> - **Temperature**: 0.6–0.8
> - **Top-p**: 0.9–0.95
> - **Prompt format**: Wrap your inputs in `system` + `user` messages if using chat models.


## ⚙️ Optional: Customize Settings for Best Results

In the model chat view, click the ⚙️ gear icon to adjust generation parameters:

- **Temperature**: Controls randomness. Try `0.7`.
- **Top-k / Top-p**: Set `top_k=50`, `top_p=0.9` for more diverse outputs.
- **Max Tokens**: Increase if responses are getting cut off (e.g. `256` or `512`).
- **Stop Sequences**: Add custom stop tokens (e.g., `</s>`) if needed.

## 🚀 Performance Tips

- Use **Q4_K_M** or **Q6_K** quantization for speed/accuracy tradeoff.
- On Mac: LM Studio uses **Metal** acceleration.
- On Windows/Linux: Enable **GPU acceleration (CUDA)** in LM Studio settings.
- Disable streaming if you want full responses faster.

If the model doesn’t start, double-check:
- GGUF format version
- Model file is complete and not corrupted

## 🧯 Troubleshooting

**Model fails to load or crashes on startup?**
- Check that your `.gguf` file is valid and complete.
- Make sure you're using a **supported quantization type** (e.g. `Q4_K_M`, `Q5_K_M`, `Q6_K`).
- Restart LM Studio after importing large models.

**Responses are garbled or incoherent?**
- Try a **new prompt template** or reformat into a chat-style message:
  ```json
  [
    {"role": "system", "content": "You are a clinical coding assistant."},
    {"role": "user", "content": "Patient has elevated blood glucose. What ICD-10 codes apply?"}
  ]

