import copy
import os
import tempfile
import torch
from datasets import load_dataset
from PIL import Image

from llama_recipes.diffag.diffag_utils import load_sd_attn  # Assumed available

# Change cache and temp directories
tempfile.tempdir = "/home/jovyan/vol-1/root_chen/git_chen/data/tmp"
# os.environ["HF_DATASETS_CACHE"] = "/home/jovyan/vol-1/root_chen/git_chen/data/tmp/alternative_cache"

def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i:i+3] == target:
            seq[i], seq[i+1], seq[i+2] = -100, -100, -100
    return seq

def tokenize_dialogs(dialogs, images, processor, dataset_config):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [j for j, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # Headers: system "<|start_header_id|>system<|end_header_id|>" and user "<|start_header_id|>user<|end_header_id|>"
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for idx in eot_indices:
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs, current_seq):
                labels[last_idx:idx+1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        # Mask the assistant header "<|start_header_id|>assistant<|end_header_id|>"
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token (128256)
        for j in range(len(labels)):
            if labels[j] == processor.tokenizer.pad_token_id or labels[j] == 128256:
                labels[j] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    batch["rgb_img"] = images
    return batch

def transform_sample(sample):
    """
    Convert a single-round VQA sample from the new JSON format to the internal format.
    Instead of loading the image now, just store the image path.
    """
    # Set index: if id is not a digit, default to 0
    sample["index"] = int(sample["id"]) if sample["id"].isdigit() else 0

    # Instead of loading the image with PIL here, store the path for lazy loading later.
    sample["image_path"] = sample["image"]

    # Convert conversations to expected "texts" structure
    convs = sample.get("conversations", [])
    if len(convs) != 2:
        raise ValueError(f"Sample id {sample.get('id', 'N/A')} should have exactly one user and one assistant conversation; got {len(convs)}")
    sample["texts"] = [{
        "user": convs[0]["content"].strip(),
        "assistant": convs[1]["content"].strip()
    }]
    return sample

def get_custom_dataset(dataset_config, processor, split="train"):
    """
    Load the single-round VQA dataset from a JSON file specified in dataset_config.data_path.
    No explicit split is applied.
    """
    # Use a custom key like "train" instead of the reserved "all"
    dataset_dict = load_dataset("json", data_files={"train": dataset_config.data_path})
    dataset = dataset_dict["train"]
    
    # Transform each sample (this now only adds meta-data without loading images)
    dataset = dataset.map(transform_sample, load_from_cache_file=False)
    return dataset

class DataCollator:
    def __init__(self, processor, dataset_config):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"  # Use right padding during training
        self.dataset_config = dataset_config

    def __call__(self, samples):
        dialogs, images, sample_indices, sd_attn_list = [], [], [], []
        for sample in samples:
            # Lazy load the image from the stored file path at batch time.
            image_path = sample["image_path"]
            image = Image.open(image_path).convert("RGB")
            
            sample_list = sample["texts"]
            dialog = []
            # Since there is only one conversation round, we follow the same structure.
            for i, sample_dict in enumerate(sample_list):
                if i == 0:
                    # For the first conversation round, attach the image token.
                    dialog += [
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample_dict["user"]}]},
                        {"role": "assistant", "content": [{"type": "text", "text": sample_dict["assistant"]}]}
                    ]
                else:
                    dialog += [
                        {"role": "user", "content": [{"type": "text", "text": sample_dict["user"]}]},
                        {"role": "assistant", "content": [{"type": "text", "text": sample_dict["assistant"]}]}
                    ]
            dialogs.append(dialog)
            images.append([image])
            sample_indices.append(sample["index"])

            if self.dataset_config.image_train_attn_path != "":
                img_idx = sample["index"]
                imgname = f"image_{img_idx}.jpg"
                if os.path.exists(os.path.join(self.dataset_config.image_train_attn_path, imgname, "lpm_attention")):
                    # Load attention maps if available.
                    sd_attn, idx = load_sd_attn(self.dataset_config, imgname, img_idx)
                    sd_attn_list.append(sd_attn)

        batch = tokenize_dialogs(dialogs, images, self.processor, self.dataset_config)
        batch["sd_attn"] = sd_attn_list
        return batch

def get_data_collator(processor, dataset_config):
    return DataCollator(processor, dataset_config)
