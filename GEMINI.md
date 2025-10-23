# nanochat

## Project Overview

This repository contains `nanochat`, a full-stack implementation of a ChatGPT-like Large Language Model (LLM). The project has been modified to use an image-based input instead of a text-based one. This change was made to improve information compression, allow for a more general information stream, and enable the use of bidirectional attention by default.

The core of the project is a GPT model implemented in PyTorch, which now includes a Vision Transformer (ViT) to process image-based inputs. The tokenizer has been removed, and the model now uses bidirectional attention by default. The project also includes a suite of scripts for training and evaluating the model, as well as a simple web UI for interacting with the trained model.

## Building and Running

The project uses `uv` for package management and `torchrun` for distributed training. The main script for running the entire pipeline is `speedrun.sh`.

### Quick Start

1.  **Set up the environment:**
    ```bash
    # Install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Create a virtual environment
    uv venv
    # Install dependencies
    uv sync --extra gpu
    # Activate the virtual environment
    source .venv/bin/activate
    ```

2.  **Run the speedrun script:**
    ```bash
    bash speedrun.sh
    ```
    This script will download the dataset, pretrain the model, and run evaluations. The entire process takes about 4 hours on an 8XH100 node.

3.  **Chat with the model:**
    Once the `speedrun.sh` script has finished, you can chat with the model via a web UI:
    ```bash
    python -m scripts.chat_web
    ```
    You can also chat with the model via the command line:
    ```bash
    python -m scripts.chat_cli -p "Why is the sky blue?"
    ```

### Running on CPU

The project can also be run on a CPU, although it will be much slower. The `dev/runcpu.sh` script provides an example of how to run the model on a CPU.

## Development Conventions

*   **Configuration:** The project uses a custom configuration system in `nanochat/configurator.py`. Configuration can be overridden via a configuration file or command-line arguments.
*   **Logging:** The project uses `wandb` for logging, but it is optional. To use `wandb`, you need to log in to your `wandb` account and set the `WANDB_RUN` environment variable.
*   **Testing:** The project has a small suite of tests that can be run with `pytest`.
*   **Reporting:** The project includes a `report` module that generates a markdown report of the training and evaluation results. The report is saved in the `report.md` file.

## Development Workflow

I will now update the local git repository after each iteration and commit to master when asked.