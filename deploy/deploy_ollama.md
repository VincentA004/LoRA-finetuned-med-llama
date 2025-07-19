# Deploying Your GGUF Model in Ollama

This guide explains how to run your quantized GGUF model in **Ollama** — a fast, local LLM runtime with GPU support and built-in REST API access.

Ollama is ideal for:
-   Private desktop inference (Mac/Linux/WSL)
-   Easy scripting via HTTP API
-   Integrations with apps like Obsidian, VSCode, and LangChain

---
## 🛠 Requirements

-   A quantized GGUF file (e.g., `llama-3-med-Q4_K_M.gguf`)
-   Ollama installed: https://ollama.com/download
-   Enough disk space (5–10 GB recommended)

> Ollama currently supports Linux, macOS, and WSL2 on Windows.

### Step 1: Create a working directory

Open a terminal and create a new folder to hold your Ollama model:

```bash
mkdir ollama-med-llama

cd ollama-med-llama
```

### Step 2: Add your GGUF model file

Copy your quantized model file into this directory.

```bash
# Replace the source path with the actual path to your model file
cp /path/to/your/project/models/gguf/llama3-med.Q4_K_M.gguf .
```

### Step 3: Create a `Modelfile`

A **`Modelfile`** is a blueprint that tells Ollama how to run your model. Create a file named `Modelfile` in your directory.

```bash
touch Modelfile
```


Now, open the Modelfile and add the following content. Make sure the FROM line exactly matches your .gguf filename.

```
# The GGUF file to use as the base model
FROM ./llama3-med.Q4_K_M.gguf

# Set the prompt template for Llama 3 Instruct
TEMPLATE """<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Set a default system message
SYSTEM """You are a helpful medical coding assistant. Respond with relevant ICD-10 or CPT codes based on the clinical notes provided. Be concise and accurate."""

# Set default generation parameters
PARAMETER temperature 0.7
PARAMETER top_k 50
PARAMETER top_p 0.9
```

### Step 4: Create the Ollama Model

Use the `Modelfile` to create a named model in Ollama's local registry. We'll call it `med-llama`.

```bash
ollama create med-llama -f ./Modelfile
```

### Step 5: Run the Model

You can now interact with your model in two ways:

#### 1. Via the Command Line

Run the model directly in your terminal for an interactive chat session:

```bash
ollama run med-llama
```

Once the model loads, you can type your prompt:
> Patient presents with chest pain and shortness of breath. What are likely ICD-10 codes?

### Via the REST API

Ollama automatically exposes a REST API on port `11434`. You can send requests to it using tools like `curl`:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "med-llama",
  "prompt": "Patient has elevated blood glucose. What ICD-10 codes apply?",
  "stream": false
}'
