# Diffusion Instruction Tuning by chen.jin@astrazeneca.com
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from typing import Dict
import wandb

from copy import deepcopy
from collections import defaultdict
import torch
import spacy
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torchvision.transforms import Resize, ToTensor
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
# Define paths for the nltk resources
# Define paths for the NLTK resources
punkt_path = '/root/nltk_data/tokenizers/punkt_tab/portuguese'
tagger_path = '/root/nltk_data/taggers/averaged_perceptron_tagger_eng'

# Download necessary resources if they don't exist
def download_nltk_resources():
    if not os.path.exists(punkt_path):
        nltk.download('punkt_tab')
    if not os.path.exists(tagger_path):
        nltk.download('averaged_perceptron_tagger_eng')


def move_sd_attn_to_device(batch, device_name):
    """
    Moves each tensor in batch["sd_attn"] to the specified device.

    Args:
        batch (dict): Batch dictionary containing "sd_attn" as a list of dictionaries.
        device_name (str or torch.device): The device to move tensors to (e.g., "cpu", "cuda", or "xpu").

    Returns:
        None: Modifies batch["sd_attn"] in-place.
    """
    for entry in batch.get("sd_attn", []):  # Ensure "sd_attn" exists in batch
        for sub_key in entry:
            entry[sub_key] = entry[sub_key].detach().clone().to(device_name)

def extract_before_rejected_answer(conversations, max_length=3):
    # Find the index of 'What was the rejected answer?'
    if 'What was the rejected answer?' in conversations:
        index = conversations.index('What was the rejected answer?')
        # Extract the sublist containing sentences before that
        sublist = conversations[:index]
    else:
        sublist = conversations  # Return full list if the phrase is not found
    
    # Limit the sublist to a maximum length of 3
    return sublist[:max_length]


# Load the spaCy model (make sure you've installed the 'en_core_web_sm' model)
nlp = spacy.load('en_core_web_sm')

def find_subject_and_object_in_sentences(sentences):
    subject_object_list = []

    # Iterate over each sentence in the list
    for sentence in sentences:
        # Parse the sentence using spaCy
        doc = nlp(sentence)

        subject = None
        direct_object = None

        # Iterate over tokens in the sentence
        for token in doc:
            # Check if the token is the subject (nsubj or nsubjpass)
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject = token.text
                subject_object_list.append(subject)
            
            # Check if the token is the direct object (dobj)
            if token.dep_ == 'dobj':
                direct_object = token.text
                subject_object_list.append(direct_object)

    return subject_object_list

# Function to download NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        print(f"Error downloading 'averaged_perceptron_tagger': {e}")
    
    try:
        nltk.download('punkt')
    except Exception as e:
        print(f"Error downloading 'punkt': {e}")

# Call the function to download data
download_nltk_data()

def is_noun(word):
    # Tokenize the word (or sentence)
    tokens = word_tokenize(word)
    
    # Ensure that tokens are not empty
    if tokens:
        # Get the POS tag
        pos_tags = pos_tag(tokens)
        
        # Check if the POS tag is a noun (NN, NNS, NNP, NNPS are common noun tags)
        return pos_tags[0][1] in ('NN', 'NNS', 'NNP', 'NNPS')
    return False  # Return False if tokens are empty

def is_adj(word):
    # Tokenize the word (or sentence)
    tokens = word_tokenize(word)
    
    # Ensure that tokens are not empty
    if tokens:
        # Get the POS tag
        pos_tags = pos_tag(tokens)
        
        # Check if the POS tag is an adjective (JJ, JJR, JJS are common adjective tags)
        return pos_tags[0][1] in ('JJ', 'JJR', 'JJS')
    return False  # Return False if tokens are empty

def is_nav(word):
    """
    Check if the word is a noun, adjective, or verb.
    """
    # Tokenize the word
    tokens = word_tokenize(word)

    # Ensure that tokens are not empty
    if tokens:
        # Get the POS tag
        pos_tags = pos_tag(tokens)
        # Check if the POS tag is a noun (NN, NNS, NNP, NNPS), adjective (JJ, JJR, JJS), or verb (VB, VBD, VBG, VBN, VBP, VBZ)
        return pos_tags[0][1] in ('NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                                'JJ', 'JJR', 'JJS',           # Adjectives
                                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')  # Verbs
    return False  # Return False if tokens are empty

def is_punctuation(word):
    # Define common punctuation symbols
    punctuation_symbols = ['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '"', "'"]
    
    # Check if the word is one of the punctuation symbols
    return word in punctuation_symbols

def save_combined_images(rgb_img, bs, sd_attn, attn_dict, attn_keys_save, global_step, output_dir, img_size=96, attn_c_plot_list=None):
    """
    Save combined images of RGB input, SD attention, and LLM cross-attention for each batch and attention key.

    Args:
        rgb_img: raw rgb image returned by dataloader.
        bs (int): Batch size.
        sd_attn (list of dict): SD attention values.
        attn_dict (list of dict): LLM attention values.
        attn_keys_save (list of lists): List of lists, where each sublist contains the attention keys for each batch.
        global_step: Training step.
        output_dir (str): Directory to save the output images.
        img_size (int, optional): Image size to resize to. Defaults to 96.
        attn_c_plot_list (list of dict, optional): LLM attention values before conv projection.
    """

    # Loop over each batch
    for b in range(bs):
        # Convert the RGB image for batch `b` to a PIL image
        rgb_img_b = rgb_img[b][0] if isinstance(rgb_img[b], list) else rgb_img[b]  # Assuming the image is already in PIL format (from Image.open)
        rgb_img_resized = rgb_img_b.resize((img_size, img_size))

        # Initialize total width with the width of the RGB image
        total_width = rgb_img_resized.width

        # Prepare the list to store resized SD and cross-attention images
        sd_and_f_imgs_resized = []

        # Loop over each attention key in the current batch
        for attn_key in attn_keys_save[b]:
            # Convert SD attention tensor to grayscale image and resize
            sd_img = tensor_to_grayscale_image(sd_attn[b][attn_key])
            sd_img_resized = sd_img.resize((img_size, img_size))
            sd_and_f_imgs_resized.append((f'SD-{attn_key}', sd_img_resized))
            total_width += sd_img_resized.width

            # Convert cross-attention tensor to grayscale image and resize
            f_img = tensor_to_grayscale_image(attn_dict[b][attn_key])
            f_img_resized = f_img.resize((img_size, img_size))
            sd_and_f_imgs_resized.append((f"Attn proj-{attn_key}", f_img_resized))
            total_width += f_img_resized.width

            # optional if attn_c_plot_list is not none, plot raw attn before projection
            if attn_c_plot_list is not None:
                f_img_raw = tensor_to_grayscale_image(attn_c_plot_list[b][attn_key])
                f_img_raw_resized = f_img_raw.resize((img_size, img_size))
                sd_and_f_imgs_resized.append((f"Attn raw-{attn_key}", f_img_raw_resized))
                total_width += f_img_raw_resized.width

        # Create a new combined image with enough width to fit RGB, SD, and all attention images
        combined_img = Image.new('RGB', (total_width, rgb_img_resized.height + 20))  # Add space for labels

        # Paste the resized RGB image into the combined image
        combined_img.paste(rgb_img_resized, (0, 0))

        # Paste each resized SD and cross-attention image into the combined image
        current_x_offset = rgb_img_resized.width
        for label, img_resized in sd_and_f_imgs_resized:
            combined_img.paste(img_resized, (current_x_offset, 0))
            current_x_offset += img_resized.width

        # Draw the labels below each image
        draw = ImageDraw.Draw(combined_img)
        font = ImageFont.load_default()  # Use the default font
        text_color = (255, 255, 255)  # White color

        # Label positions
        label_offset = rgb_img_resized.height + 5  # Adjust the label height
        hor_offset = 0

        # Add the label for the RGB image
        draw.text((hor_offset + rgb_img_resized.width // (len(sd_and_f_imgs_resized) + 2), label_offset), "Input Img", font=font, fill=text_color)

        # Add labels for each SD and LLM attention image
        current_x_offset = rgb_img_resized.width
        for label, img_resized in sd_and_f_imgs_resized:
            draw.text((hor_offset + current_x_offset + img_resized.width // (len(sd_and_f_imgs_resized) + 2), label_offset), label, font=font, fill=text_color)
            current_x_offset += img_resized.width

        # Ensure the output directory exists and save the combined image
        os.makedirs(output_dir, exist_ok=True)
        # Combine all words in attn_keys_save[b] separated by underscores
        attn_keys_str = "_".join(attn_keys_save[b])
        # Save the image with the combined attention keys in the filename
        combined_img.save(f"{output_dir}/Fattn_vs_SDattn_step_{global_step}_{attn_keys_str}.png")
    return combined_img

def extract_masks(data_dict, vllm_tensor):
    """
    Extract vision and text masks from the data_dict data based on image boundaries and attention masks.

    Args:
        data_dict (dict): A dictionary containing data_dict data with keys 'image_bound' and 'attention_mask'.
        vllm_tensor (torch.Tensor): The embedding tensor from which the masks will be created.

    Returns:
        text_mask (torch.Tensor): A mask indicating the positions of text tokens.
        vision_mask (torch.Tensor): A mask indicating the positions of vision tokens.
    """
    # Initialize masks with appropriate sizes
    vision_mask = torch.zeros(vllm_tensor.size(0), vllm_tensor.size(1), dtype=torch.bool, device=vllm_tensor.device)
    text_mask = torch.ones(vllm_tensor.size(0), vllm_tensor.size(1), dtype=torch.bool, device=vllm_tensor.device)

    # Update the masks based on image boundaries
    for i in range(len(data_dict['image_bound'])):
        image_bound = data_dict['image_bound'][i]
        if len(image_bound) > 0:
            for bound in image_bound:
                vision_mask[i, bound[0]:bound[1]] = True
                text_mask[i, bound[0]:bound[1]] = False

        # Check if the length of attention_mask[i] is less than the length of text_mask[i]
        if len(data_dict['attention_mask'][i]) < len(text_mask[i]):
            # Pad attention_mask[i] with False on the right to match the length of text_mask[i]
            padding_length = len(text_mask[i]) - len(data_dict['attention_mask'][i])
            padding = torch.zeros(padding_length, dtype=torch.bool, device=text_mask.device)
            padded_attention_mask = torch.cat((data_dict['attention_mask'][i].bool(), padding), dim=0)
        elif len(text_mask[i]) < len(data_dict['attention_mask'][i]):
            padded_attention_mask = data_dict['attention_mask'][i][:len(text_mask[i])].bool()
        else:
            padded_attention_mask = data_dict['attention_mask'][i].bool()

        # Refine the text_mask using the attention_mask to exclude padding
        text_mask[i] = text_mask[i] & padded_attention_mask

    return text_mask, vision_mask

def load_sd_attn(self, imgname: str, idx: int) -> Dict[str, torch.Tensor]:
    """
    Loads and processes spatial attention maps for a given image.

    This function searches for attention map images within subdirectories of the specified 
    `image_train_attn_path`, converts them to tensors, and stores them in a dictionary 
    with words as keys and tensors as values.

    Args:
        self: Instance of the class.
        imgname (str): Path or name of the image for which attention maps are being loaded.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping words to their corresponding attention map tensors.

    Notes:
        - If `image_train_attn_path` is empty or no attention maps are found, the function fetches the next sample.
    """
    if '/' in imgname:
        imgname = imgname.split('/')[-1]
    
    attn_path = os.path.join(
        self.image_train_attn_path,
        imgname,
        "lpm_attention",
    )
    # Initialize a dictionary to hold the word and corresponding tensor
    sd_attn = {}

    # Check if attn_path exists
    if os.path.exists(attn_path):
        # Find all subdirectories in attn_path
        subdirectories = [
            d for d in os.listdir(attn_path)
            if os.path.isdir(os.path.join(attn_path, d))
        ]

        # Iterate over subdirectories and load attention tensors
        for subdir in subdirectories:
            # Construct the pattern to match the specific image files
            pattern = os.path.join(attn_path, subdir, "attention_*.jpg")
            # Use glob to find all files matching the pattern
            for image_path in glob.glob(pattern):
                # Extract the word from the filename
                filename = os.path.basename(image_path)
                # Extracts the word between 'attention_' and '.jpg'
                word = filename.split("_")[1].split(".")[0]
                # Open the image (as 8-bit grayscale) and convert to tensor
                image = Image.open(image_path).convert("L")
                transform = ToTensor()
                image_tensor = transform(image)
                # Store the tensor in the dictionary
                sd_attn[word] = image_tensor.squeeze(0)
    # else:
    #     # If attn_path does not exist, get the next sample
    #     idx += 1
    #     if idx >= len(self.raw_data):
    #         idx = 0
    #     return self.__getitem__(idx), idx

    # if not sd_attn:
    #     print("sd_attn is empty, get the next sample ...")
    #     idx += 1
    #     if idx >= len(self.raw_data):
    #         idx = 0
    #     return self.__getitem__(idx), idx

    return sd_attn, idx

def tensor_to_grayscale_image(sd_attn):
    """
    Converts a tensor (shape: [32, 32], range: [0, 1]) to a grayscale image using PIL.

    Args:
        sd_attn: A PyTorch tensor representing the attention map.

    Returns:
        A PIL Image object representing the grayscale image.
    """

    # Ensure the tensor is on CPU and has float32 data type
    sd_attn_c = sd_attn.clone().cpu().detach().float()

    # Scale the values to the 0-255 range for image representation (optional)
    # If your tensor values are already between 0 and 1, you can comment out this line.
    sd_attn_c = sd_attn_c * 255

    # Convert the tensor to a NumPy array
    sd_attn_np = sd_attn_c.numpy().astype(np.uint8)

    # Convert the NumPy array to a grayscale image using PIL
    image = Image.fromarray(sd_attn_np, "L")  # 'L' mode for grayscale

    return image

def exponential_decay_scheduler(
    MSE_weight, current_step, total_step, factor=1.0, inverse=False
):
    """
    Calculate the scaling factor for the MSE loss weight using an adjustable exponential decay,
    with an option to invert the decay progression.

    Args:
    MSE_weight (float): The initial MSE weight at the beginning of training.
    current_step (int): Current training step.
    total_step (int): Total number of training steps.
    factor (float): Factor to adjust the steepness of the decay.
    inverse (bool): If True, inverse the decay progression.

    Returns:
    float: Scaled MSE weight for the current step.
    """
    # Calculate the base decay rate
    base_decay_rate = -1.0 / total_step

    # Apply the factor to adjust the decay rate
    adjusted_decay_rate = base_decay_rate * factor

    # Calculate the scaling factor
    scaling_factor = MSE_weight * np.exp(adjusted_decay_rate * current_step)

    # Invert the decay rate if the inverse flag is true
    if inverse:
        scaling_factor = 1 - scaling_factor  # Inverting the rate

    return scaling_factor