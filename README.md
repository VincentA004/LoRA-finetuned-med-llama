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

## Available Commands

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
