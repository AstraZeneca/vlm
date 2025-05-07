# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools
import torch

from llama_recipes.diffag.diffag_utils import *
import tempfile
import os
# Change cache and temp directories
tempfile.tempdir = "/home/jovyan/vol-1/root_chen/git_chen/data/tmp"
# os.environ["HF_DATASETS_CACHE"] = "/home/jovyan/vol-1/root_chen/git_chen/data/tmp/alternative_cache"

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq
def tokenize_dialogs(dialogs, images, processor, dataset_config):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    batch["rgb_img"] = images
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    # Set the environment variable to point to the cache directory
    os.environ["HF_DATASETS_CACHE"] = "/root/local_hg_datasets"
    dataset_dict = load_dataset("HuggingFaceM4/the_cauldron", name="ocrvqa")
    dataset = dataset_dict['train']
    dataset = dataset.select(range(30000))

    # Use dataset.map with a lambda to add an absolute index
    dataset = dataset.map(lambda x, idx: {"index": idx}, with_indices=True, batched=True, load_from_cache_file=False)

    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]
    return dataset

class OCRVQADataCollator:
    def __init__(self, processor,dataset_config):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
        self.dataset_config = dataset_config
    def __call__(self, samples):
        dialogs, images, sample_indices, sd_attn_list = [], [], [], []
        for sample in samples:
            image_list,sample_list = sample["images"],sample["texts"]
            if len(image_list) > 1:
                raise ValueError("Only support one image per sample")
            image = image_list[0].convert("RGB") # only use the first image
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    # only append image to the first sentence
                    dialog += [
                    {"role":"user","content":[{"type": "image"},{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
                
                else:
                    dialog += [
                    {"role":"user","content":[{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
            dialogs.append(dialog)
            images.append([image])

            sample_indices.append(sample["index"])  # Access the absolute index added during dataset.map

            if self.dataset_config.image_train_attn_path != "": 
                img_idx = sample["index"]
                imgname = f"image_{img_idx}.jpg"
                if os.path.exists(os.path.join(self.dataset_config.image_train_attn_path, imgname, "lpm_attention")):
                    # sd_attn - Dict[str, torch.Tensor]: A dictionary mapping words to their corresponding attention map tensors.
                    sd_attn, idx = load_sd_attn(self.dataset_config, imgname, img_idx)
                    sd_attn_list.append(sd_attn)

        # Tokenize dialogs and images
        batch = tokenize_dialogs(dialogs, images, self.processor, self.dataset_config)
        # batch["sample_indices"] = torch.tensor(sample_indices)  # Include the original indices in the batch
        batch["sd_attn"] = sd_attn_list  # Include the batch (list) of sd_attn each is a Dict['word': sd_attn_map]

        return batch
def get_data_collator(processor,dataset_config):
    return OCRVQADataCollator(processor,dataset_config)
