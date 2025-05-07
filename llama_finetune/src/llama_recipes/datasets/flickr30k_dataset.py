# Diffusion Instruction Tuning by chen.jin@astrazeneca.com
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import torch
from datasets import Dataset
from transformers import AutoProcessor
from llama_recipes.diffag.diffag_utils import *
import json
from PIL import Image

def get_custom_dataset(dataset_config, processor, split="train", split_ratio=0.9):
    # Load and format the Flickr30k dataset JSON
    raw_data = json.load(open(dataset_config.data_path, "r"))["images"]
    if dataset_config.length_train_eval == -1:
        total_len = len(raw_data)
        split_idx = int(total_len * split_ratio)
        data = raw_data[:split_idx] if split == 'train' else raw_data[split_idx:]
    else:
        train_idx_start = dataset_config.length_train_eval * dataset_config.multi_stage_idx
        train_idx_end = dataset_config.length_train_eval * (dataset_config.multi_stage_idx + 1)
        eval_idx = 0 - dataset_config.length_train_eval
        
        # Wrap only train indices if they exceed raw_data's length
        train_idx_start %= (len(raw_data) - dataset_config.length_train_eval)
        train_idx_end %= (len(raw_data) - dataset_config.length_train_eval)

        data = raw_data[train_idx_start:train_idx_end] if split == 'train' else raw_data[eval_idx:]

    # Prepare the data in the required dialog format
    formatted_data = []
    skip_count = 0
    for idx, sample in enumerate(data):
        if dataset_config.image_train_attn_path:
            image_path = sample["filename"]
            imgname = os.path.basename(image_path)
            attn_path = os.path.join(dataset_config.image_train_attn_path, imgname, "lpm_attention")
            if not os.path.exists(attn_path) or not os.listdir(os.path.join(attn_path, imgname.split('.')[0])):
                # print(f"The directory '{attn_path}' does not exist or is empty. Skip {skip_count}")
                skip_count += 1
                continue

        image_path = os.path.join("/home/jovyan/vol-1/root_chen/git_chen/data/data_chen/openflamingo_data/flickr30k-images", sample["filename"])
        captions = sample["sentences"]

        # Use only the first caption if the flag is set
        if dataset_config.use_first_caption_only:
            captions = captions[:1]

        # Set the user question prompt
        user_prompt = "Describe the image in a single sentence as a caption."

        for caption in captions:
            formatted_dialog = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption["raw"].strip()}]
                }
            ]

            formatted_data.append({
                "dialog": formatted_dialog,
                "image": image_path,
                "index": idx
            })

    if dataset_config.image_train_attn_path:
        print(f"Skipped {skip_count} data who do not have sdattn")
    # Convert to Hugging Face Dataset format and add index column
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(lambda x, idx: {"index": idx}, with_indices=True)

    return dataset

class Flickr30kDataCollator:
    def __init__(self, processor, dataset_config):
        self.processor = processor
        self.dataset_config = dataset_config
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, samples):
        dialogs = [sample["dialog"] for sample in samples]
        # Open, resize, and convert images
        images = [Image.open(sample["image"]).convert("RGB").resize((560, 560)) for sample in samples]

        # Load `sd_attn` dynamically if required
        sd_attn_list = []
        if self.dataset_config.image_train_attn_path:
            for sample in samples:
                idx = sample["index"]
                image_path = sample["image"]
                imgname = os.path.basename(image_path)
                attn_path = os.path.join(self.dataset_config.image_train_attn_path, imgname, "lpm_attention")

                if os.path.exists(attn_path):
                    sd_attn, _ = load_sd_attn(self.dataset_config, imgname, idx)
                    sd_attn_list.append(sd_attn)
                else:
                    sd_attn_list.append({'nosdatnplaceholder': None})

        # Tokenize the dialogs
        text_prompts = self.processor.apply_chat_template(dialogs)
        batch = self.processor(images=images, text=text_prompts, padding=True, return_tensors="pt")

        # Generate labels
        label_list = []
        for i in range(len(batch["input_ids"])):
            dialog_tokens = batch["input_ids"][i].tolist()
            labels = dialog_tokens.copy()

            eot_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            eot_indices = [idx for idx, token in enumerate(labels) if token == eot_token_id]

            # Define header sequences
            prompt_headers = [
                self.processor.tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "system", "<|end_header_id|>"]),
                self.processor.tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "user", "<|end_header_id|>"])
            ]
            assistant_header = self.processor.tokenizer.convert_tokens_to_ids(["<|start_header_id|>", "assistant", "<|end_header_id|>"])

            # Mask prompt headers and pad tokens
            last_idx = 0
            for idx in eot_indices:
                current_seq = labels[last_idx:idx + 1]
                if any(header in current_seq for header in prompt_headers):
                    labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
                else:
                    last_idx = idx + 1

            labels = self.replace_header_sequence(assistant_header, labels)

            # Mask padding and image tokens
            for j, token_id in enumerate(labels):
                if token_id == self.processor.tokenizer.pad_token_id or token_id == self.processor.image_token_id:
                    labels[j] = -100

            label_list.append(labels)

        # Create the batch
        batch["labels"] = torch.tensor(label_list)
        batch["rgb_img"] = images
        if self.dataset_config.image_train_attn_path:
            batch["sd_attn"] = sd_attn_list if sd_attn_list else [{'nosdatnplaceholder': None}]

        return batch

    def replace_header_sequence(self, header_sequence, labels):
        seq_len = len(header_sequence)
        for i in range(len(labels) - seq_len + 1):
            if labels[i:i + seq_len] == header_sequence:
                labels[i:i + seq_len] = [-100] * seq_len
        return labels

# The function to return the collator for Flickr30k
def get_data_collator(processor, dataset_config):
    return Flickr30kDataCollator(processor, dataset_config)