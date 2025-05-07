# Llama Finetune Example

This subfolder contains a minimal example for fine-tuning Llama-style models, adapted from the [`llama-recipes`](https://github.com/meta-llama/llama-cookbook/tree/21647d42a35072522d46b10da9c3df1b0326a26f) repository, for more details please refer to [Llama-Recipes Quickstart](https://github.com/meta-llama/llama-cookbook/tree/21647d42a35072522d46b10da9c3df1b0326a26f/recipes/quickstart).

## Structure

-   `src/llama_recipes/`: Contains the core library code for fine-tuning, including:
    -   `configs/`: Configuration files for training, datasets, FSDP, etc.
    -   `datasets/`: Various dataset loading and processing utilities (including the user-highlighted `datasets` module).
    -   `diffag/`: Code related to the Lavender method, which focuses on creating wrapper functions around models of interest to catch and modify cross-attention operations.
    -   `mllama/`: Code related to the Llama-3.2 model.
    -   `policies/`: Policies for mixed precision, activation checkpointing, etc.
    -   `utils/`: Utility functions for training, FSDP, memory, etc.
    -   `model_checkpointing/`: Utilities for handling model checkpoints.
    -   `finetuning.py`: The main script for running the fine-tuning process.
-   `examples/`: Contains example scripts to launch fine-tuning.
    -   `run_finetune.sh`: A sample shell script to demonstrate how to run `finetuning.py`. You will need to adapt the parameters within this script (e.g., model name, dataset path, output directory).
-   `data/`: Contains sample data for the fine-tuning example.
    -   `sample_dataset.jsonl`: A small example dataset in JSON Lines format.
-   `requirements.txt`: A list of Python dependencies required to run the fine-tuning example.

## Setup

1.  **Install Dependencies:**
    Navigate to this directory (`vlm/llama_finetune/`) and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    You may also need to install a spacy model if you haven't already:
    ```bash
    python -m spacy download en_core_web_sm
    ```

2.  **Prepare Your Model and Data:**
    -   Ensure you have the base model you intend to fine-tune (e.g., a Llama-3.2 compatible model). You will need to provide the path to this model or its Hugging Face identifier.
    -   Download dataset following [Attention Alignment Data](../readme.md#3-attention-alignment-data)
    -   Prepare your custom dataset in a format compatible with the scripts (e.g., JSON Lines, as shown in `data/sample_dataset.jsonl`). Update the dataset path in the launch script.

## Running the Example

1.  Navigate to the `examples` directory:
    ```bash
    cd examples
    ```
2.  Modify the `run_finetune.sh` script:
    -   Set `model_name` to your base model.
    -   Update `custom_dataset.file` to point to your dataset.
    -   Adjust other parameters like `output_dir`, `batch_size_training`, `num_epochs`, etc., as needed.
3.  Execute the script:
    ```bash
    bash run_finetune.sh
    ```

**Note on GPU Usage:** This example assumes that a compatible GPU environment is available for fine-tuning. The scripts are designed to leverage GPU acceleration. If you are running in an environment without a GPU, the training process will likely fail or be extremely slow. The parts of the code requiring GPU execution are included, assuming the user has the necessary hardware.

## Anonymization

Any specific paths or user identifiers have been removed or replaced with generic placeholders (e.g., `<your_model_name_or_path>`, `./data/sample_dataset.jsonl`). Please ensure you update these placeholders with your actual paths and configurations.



## License
This repository and all associated code are licensed under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) License. 
Lavender finetuned models are also under their original base model licences: 

- Lavender-Llama-3.2-11B-Lora & Lavender-Llama-3.2-11B-Full are derived from Llama-3.2-11B-Vision-Instruct and is originally licensed under [llama3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt).
- Lavender-MiniCPMv2.5-Lora & Lavender-MiniCPMv2.5-Full are derived from 
MiniCPM-Llama3-V-2_5, which is derived from Llama3 and licensed under [llama3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).

Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.

## Citation

If you make use of our work, please cite our paper:

```
@misc{jin2025diffusioninstructiontuning,
    title={Diffusion Instruction Tuning}, 
    author={Chen Jin and Ryutaro Tanno and Amrutha Saseendran and Tom Diethe and Philip Teare},
    year={2025},
    eprint={2502.06814},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2502.06814}, 
}
```
