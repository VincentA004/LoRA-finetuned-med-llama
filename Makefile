# Makefile for the LLaMA Fine-Tuning Project

# Use the python interpreter from the virtual environment
PYTHON := .venv/bin/python

# Add PYTHONPATH to tell Python where to find the 'src' module
PYTHON_EXEC := PYTHONPATH=. $(PYTHON)

.PHONY: all setup train export serve benchmark-onnx publish clean

# Default target
all: setup train export

# Target: setup
# Description: Creates a virtual environment and installs dependencies.
setup:
	@if [ ! -d ".venv" ]; then \
		echo ">> Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	@echo ">> Installing dependencies from requirements.txt..."
	@$(PYTHON) -m pip install -q -r requirements.txt
	@echo "✅ Environment is ready."

# Target: train
# Description: Runs the full training and merging pipeline.
train:
	@echo ">> Starting LoRA fine-tuning..."
	@$(PYTHON_EXEC) src/training/train_lora.py
	@echo ">> Merging LoRA adapter into the base model..."
	@$(PYTHON_EXEC) src/training/merge_lora.py
	@echo "✅ Training and merging complete."

# Target: export
# Description: Exports the model to ONNX and GGUF formats.
export:
	@echo ">> Exporting model to ONNX..."
	@$(PYTHON_EXEC) src/export/export_model.py --format onnx
	@echo ">> Exporting model to GGUF and quantizing to Q4_K_M..."
	@$(PYTHON_EXEC) src/export/export_model.py --format gguf --quantize Q4_K_M
	@echo "✅ Export process complete."

# Target: serve
# Description: Starts the FastAPI inference server with a specified model format.
# Usage: make serve format=safetensor  OR  make serve format=onnx
serve:
	@echo ">> Starting FastAPI server with $(format) model..."
	@$(PYTHON_EXEC) src/inference/server.py --format $(format)

# Target: benchmark-onnx
# Description: Benchmarks the ONNX model.
benchmark-onnx:
	@echo ">> Benchmarking ONNX model..."
	@$(PYTHON_EXEC) src/inference/bench_onnx.py --model $(shell $(PYTHON_EXEC) -c 'import src.config as c; print(c.ONNX_EXPORT_PATH)')

# Target: publish
# Description: Pushes all model artifacts to the Hugging Face Hub.
publish:
	@echo ">> Publishing artifacts to Hugging Face Hub..."
	@$(PYTHON_EXEC) publish/push_to_huggingface.py
	@echo "✅ Publishing complete."

# Target: clean
# Description: Removes all generated artifacts and caches.
clean:
	@echo ">> Cleaning up generated directories and files..."
	@rm -rf models
	@rm -rf .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "🧹 Workspace cleaned."