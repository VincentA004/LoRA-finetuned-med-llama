# End-to-End LoRA Fine-Tuning Pipeline for Medical LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/models)

This repository contains a complete, end-to-end MLOps pipeline for fine-tuning, evaluating, and deploying a language model for medical code prediction. The project utilizes Meta's **Llama 3 8B Instruct** model and fine-tunes it on the `FiscaAI/synth-ehr-icd10cm-prompt` dataset to assist with clinical coding tasks.

The entire workflow is automated and containerized, demonstrating best practices for creating reproducible and production-ready machine learning systems. It covers the full lifecycle from training with LoRA to exporting the model in efficient formats like GGUF and ONNX for flexible deployment.

## Key Features

-   **LoRA Fine-Tuning:** Efficiently fine-tunes the Llama 3 model using Low-Rank Adaptation (LoRA) and 4-bit quantization (QLoRA) with the `peft` and `trl` libraries.
-   **Automated Workflow:** A `Makefile` automates the entire process from setup and training to exporting and cleaning, enabling one-command execution.
-   **Multi-Format Export:** Supports exporting the fine-tuned model to both **ONNX** for cross-platform CPU/GPU inference and **GGUF** for use with `llama.cpp`.
-   **Inference & Deployment:** Includes a ready-to-use **FastAPI** server for REST API deployment and scripts for performance benchmarking.
-   **Centralized Configuration:** All paths, hyperparameters, and model names are managed in a single `src/config.py` file for easy modification and control.

## Project Quickstart

This project is structured for clarity and modularity, separating concerns like training, exporting, and inference into distinct directories. The entire workflow is managed by a `Makefile` for simple, one-command execution.

### Project Structure

### Automated Workflow with `make`

To run the pipeline, first ensure you have **Git**, **Python 3.9+**, and **make** installed.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/lora-finetuned-med-llama.git](https://github.com/your-username/lora-finetuned-med-llama.git)

cd lora-finetuned-med-llama
```
2. Set Up the Environment

This command creates a local Python virtual environment and installs all required dependencies.

```bash
make setup
```

3. Run the Full Pipeline

This single command will run the entire training and exporting pipeline sequentially. It will download the base model, fine-tune it with LoRA, merge the adapter, and then export the final model to both ONNX and GGUF formats.

```bash
make all
```

#### Available Commands

The `Makefile` provides several commands to manage the project lifecycle:

| Command                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `make setup`                  | Creates a virtual environment and installs all dependencies.                |
| `make train`                  | Runs the LoRA training script and merges the final adapter.                 |
| `make export`                 | Exports the fine-tuned model to both ONNX and GGUF formats.                |
| `make serve format=[fmt]`    | Starts the API server. Set `fmt` to `safetensor` or `onnx`.                |
| `make benchmark-onnx`        | Runs the performance benchmark on the exported ONNX model.                 |
| `make publish`               | Pushes the final model artifacts to the Hugging Face Hub.                  |
| `make clean`                 | Deletes all generated model files, caches, and the virtual environment.    |

## Technical Details

This section provides a deeper look into the libraries and techniques used to build this pipeline.

### The Hugging Face Ecosystem

The project's efficiency and power stem from the tight integration of several key Hugging Face libraries:

-   **`peft` (Parameter-Efficient Fine-Tuning)**: We employ LoRA (Low-Rank Adaptation) via `peft`. Instead of fine-tuning all 8 billion parameters, LoRA posits that the change in weights (${\Delta}W$) during adaptation has a low intrinsic rank. It therefore decomposes this change into two smaller, low-rank matrices (${\Delta}W = BA$), where only A and B are trained. This reduces the trainable parameter count by over 99%. We specifically use **QLoRA**, which further optimizes memory by loading the frozen base model in a 4-bit `NormalFloat` (NF4) data type, making it feasible to fine-tune on consumer hardware.
-   **`trl` (Transformer Reinforcement Learning)**: The `SFTTrainer` from `trl` is leveraged for Supervised Fine-Tuning. It automates the complex process of packing sequences to their maximum length, correctly formatting the conversational dataset using the Llama 3 chat template, and managing the causal language modeling training loop.
-   **`accelerate` & `bitsandbytes`**: This pair forms the backbone of the QLoRA process. `accelerate` handles abstract device placement (`device_map="auto"`) for CPU, GPU, or Apple Silicon (MPS) environments, while `bitsandbytes` provides the underlying CUDA or CPU kernels for 4-bit quantization and the memory-efficient `paged_adamw_32bit` optimizer.

### GGUF Export and Quantization

To enable high-performance inference on consumer CPUs, the model is exported to GGUF for the `llama.cpp` ecosystem.

-   **Format**: GGUF is a single-file binary format that contains the model architecture, vocabulary, and quantized weights. This self-contained design allows for extremely fast, memory-mapped model loading.
-   **Quantization Strategy**: The pipeline defaults to **`Q4_K_M`** quantization. This is a sophisticated 4.5 bits-per-weight "K-Quant" method from `llama.cpp`. Unlike older quantization schemes, K-Quants use larger block sizes (256) and improved quantization scales, resulting in significantly better perplexity (lower quality loss) for a given file size. The 'M' variant denotes a 'medium' model size and quality level within its group.

### ONNX Export for Production

For platform-agnostic and high-throughput deployment, the model is converted to the ONNX (Open Neural Network Exchange) format.

-   **Process**: The `optimum` library traces the PyTorch model's forward pass to build a static, hardware-agnostic computation graph. This decouples the model from the Python/PyTorch framework.
-   **Performance**: When executed with the **ONNX Runtime**, this static graph enables powerful optimizations not possible in a dynamic environment. Techniques like **graph fusion** (merging sequential operations like matrix multiplication and bias addition into a single kernel), constant folding, and operator elimination are applied. The runtime then targets hardware-specific execution providers (e.g., CUDA, TensorRT) for maximum performance, achieving significantly lower latency.

### Model Quantization

The pipeline supports multiple quantization strategies to optimize the model for different performance targets.

#### GGUF Quantization

The `llama.cpp` ecosystem provides a rich set of quantization methods. The desired GGUF quantization level can be easily changed by modifying the `export` command in the `Makefile` or by running the `export_model.py` script with a different `--quantize` flag.

While the default is **`Q4_K_M`**, you can generate other versions for different trade-offs:
-   **`Q8_0`**: An 8-bit quantization for near-lossless quality.
-   **`Q5_K_M`**: A 5-bit k-quant for a slight quality improvement over 4-bit.
-   **`Q2_K`**: A 2-bit k-quant for maximum compression and speed on resource-constrained devices.

#### ONNX Quantization (INT8 vs. UINT8)

ONNX quantization is typically performed as a post-export step using the **ONNX Runtime** library. The most common method is **Dynamic Quantization**, where model weights are converted to a lower precision integer format offline, while activations are quantized on-the-fly during inference.

The choice between `INT8` and `UINT8` for weights is critical:

-   **`INT8` (Signed 8-bit Integer)**: This format represents values from -128 to 127. It is the **standard and recommended choice for LLM weights**, as the weights are typically distributed symmetrically around zero (both positive and negative values). Using `INT8` accurately preserves this distribution, leading to better model performance post-quantization.

-   **`UINT8` (Unsigned 8-bit Integer)**: This format represents values from 0 to 255. It is generally used for quantizing model activations that are always non-negative (e.g., the output of a ReLU activation function). Using it for weights that are centered around zero can introduce a quantization bias and degrade model accuracy.

For this project, `INT8` dynamic quantization is the appropriate method for optimizing the ONNX model.


## Deployment

The exported GGUF and ONNX models are optimized for efficient, local inference on consumer hardware. They can be easily run using popular applications that support the `llama.cpp` ecosystem. Step-by-step guides are provided for the following services:

-   **LM Studio:** A detailed guide for running the quantized GGUF model with the user-friendly [LM Studio](https://lmstudio.ai/) desktop application.
    -   **[View the LM Studio Deployment Guide](./deploy/deploy_lmstudio.md)**

-   **Ollama:** Instructions for running the GGUF model with the fast, local [Ollama](https://ollama.com) runtime, which also provides a built-in REST API for easy integration.
    -   **[View the Ollama Deployment Guide](./deploy/deploy_ollama.md)**

## Model & Showcase

### Hugging Face Repository
All model artifacts, including the merged fine-tuned model and all ONNX / GGUF versions, are available on the Hugging Face Hub.

-   [![Hugging Face Repo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Repo-blue)](https://huggingface.co/your-hf-username/lora-finetuned-med-llama-3)
    -   **[View the Model on Hugging Face Hub](https://huggingface.co/your-hf-username/lora-finetuned-med-llama-3)**

### Project Walkthrough & Analysis
For a detailed, interactive walkthrough of the entire pipeline—including performance analysis and visualizations of the quantization trade-offs—please see the project notebook.

-   **[Open the Project Walkthrough Notebook](./walkthrough.ipynb)**
