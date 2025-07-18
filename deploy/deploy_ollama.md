# Deploying Your GGUF Model in Ollama

This guide explains how to run your quantized GGUF model in [Ollama](https://ollama.com) — a fast, local LLM runtime with GPU support and built-in REST API access.

Ollama is ideal for:
- Private desktop inference (Mac/Linux/WSL)
- Easy scripting via HTTP API
- Integrations with apps like Obsidian, VSCode, and LangChain

---

## 🛠 Requirements

- A quantized GGUF file (e.g., `llama-3-med-Q4_K_M.gguf`)
- Ollama installed: https://ollama.com/download
- Enough disk space (5–10 GB recommended)

> Ollama currently supports Linux, macOS, and WSL2 on Windows (no native Windows support yet).

### Step 1: Create a working directory

Open a terminal and create a new folder to hold your Ollama model:

```bash
mkdir ollama-med-llama
cd ollama-med-llama
```

### Step 2: Add your GGUF model file

Copy your quantized model file into this directory. For example:

