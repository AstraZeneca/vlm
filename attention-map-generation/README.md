# Attention Map Generation

This subfolder provides a minimal script for batch-extracting per-token Stable Diffusion attention maps, used as alignment targets in Lavender fine-tuning.

It is built on top of **[Local Prompt Mixing](https://github.com/orpatashnik/local-prompt-mixing)** by Patashnik et al. Please clone that repository first and install its dependencies.

## Setup

```bash
# 1. Clone the base repository
git clone https://github.com/orpatashnik/local-prompt-mixing.git
cd local-prompt-mixing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the batch extraction script into the repo
cp <path-to-this-folder>/run_seg_batch_multip.py .
```

## Input Data Format

Prepare a directory with numbered subfolders, each containing:
```
output_dir/
├── 000001/
│   ├── imgs/          # Input images (e.g., image.jpg)
│   └── last_sentence.txt   # One caption/prompt per line
├── 000002/
│   └── ...
```

## Usage

```bash
python run_seg_batch_multip.py \
    --output_dir /path/to/data \
    --start_index 1 \
    --end_index 100 \
    --plot_attn_map \
    --low_resource \
    --gpu_id 0
```

**Key arguments:**
- `--output_dir`: Root directory containing the numbered subfolders
- `--start_index` / `--end_index`: Range of subfolders to process (1-indexed)
- `--plot_attn_map`: Extract per-token attention maps (required for Lavender)
- `--low_resource`: Reduces VRAM usage
- `--local_model_path`: Optional path to a local Stable Diffusion checkpoint (defaults to `CompVis/stable-diffusion-v1-4` from HuggingFace)

## Output

Per-image attention maps are saved as grayscale JPEGs:
```
output_dir/000001/lpm_attention/<image_name>/attention_<word>.jpg
```

These are loaded during Lavender fine-tuning via the `image_train_attn_path` config parameter.
