# Batch Stable Diffusion attention map extraction for Lavender.
# Built on top of: https://github.com/orpatashnik/local-prompt-mixing
#
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import gc
import sys
import re
import glob
import argparse

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

import numpy as np
import torch
from PIL import Image

from src.attention_based_segmentation import Segmentor
from src.diffusion_model_wrapper import (
    get_stable_diffusion_model,
    get_stable_diffusion_config,
    DiffusionModelWrapper,
)
from src.null_text_inversion import invert_image
from src.prompt_to_prompt_controllers import AttentionStore


def get_sentences(subfolder_path):
    """Read unique caption lines from last_sentence.txt."""
    file_path = os.path.join(subfolder_path, "last_sentence.txt")
    sentences = set()
    with open(file_path, "r") as f:
        for line in f:
            sentence = line.rstrip()
            if sentence:
                sentences.add(sentence)
    return list(sentences)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Batch extract SD attention maps for Lavender."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory containing numbered subfolders.")
    parser.add_argument("--start_index", type=int, default=1,
                        help="Start index for subfolders (1-indexed).")
    parser.add_argument("--end_index", type=int, default=10,
                        help="End index for subfolders.")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--local_model_path", type=str, default="",
                        help="Local path to SD checkpoint. Empty = download CompVis/stable-diffusion-v1-4.")
    parser.add_argument("--auth_token", type=str, default="",
                        help="HuggingFace auth token (only needed if downloading gated models).")
    parser.add_argument("--low_resource", action="store_true",
                        help="Reduce VRAM usage.")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_num_words", type=int, default=77)
    parser.add_argument("--num_segments", type=int, default=5)
    parser.add_argument("--background_segment_threshold", type=float, default=0.35)
    parser.add_argument("--num_inner_steps", type=int, default=10)
    parser.add_argument("--plot_attn_map", action="store_true",
                        help="Extract per-token attention maps (required for Lavender).")
    parser.add_argument("--plot_attn_mask", action="store_true",
                        help="Save binary attention masks instead of grayscale.")
    parser.add_argument("--save_rec", action="store_true",
                        help="Save reconstructed images.")
    parser.add_argument("--exp_folder", type=str, default="lpm_attention",
                        help="Subfolder name for attention map output.")
    return parser


def process_subfolder(args, subfolder_path):
    """Process all images in a single subfolder."""
    special_char_pattern = re.compile(r"[^a-zA-Z0-9 ._]")
    prompts = get_sentences(subfolder_path)
    print(f"  Prompts: {prompts}")

    imgs_dir = os.path.join(subfolder_path, "imgs")
    if not os.path.isdir(imgs_dir):
        print(f"  No imgs/ directory found, skipping.")
        return

    for img_name in os.listdir(imgs_dir):
        name_base = img_name.split(".")[0]
        args.real_image_path = os.path.join(imgs_dir, img_name)
        exp_path = os.path.join(subfolder_path, args.exp_folder, name_base)

        if os.path.exists(exp_path):
            print(f"  Skipping (exists): {args.real_image_path}")
            continue

        print(f"  Processing: {args.real_image_path}")
        os.makedirs(exp_path, exist_ok=True)
        args.exp_path = exp_path

        for prompt in prompts:
            ldm_stable = get_stable_diffusion_model(args)
            ldm_stable_config = get_stable_diffusion_config(args)
            g_cpu = torch.Generator(device=ldm_stable.device).manual_seed(args.seed)

            x_t, uncond_embeddings = invert_image(
                args, ldm_stable, ldm_stable_config, [prompt], exp_path,
                save_rec=args.save_rec,
            )

            controller = AttentionStore(ldm_stable_config["low_resource"])
            wrapper = DiffusionModelWrapper(
                args, ldm_stable, ldm_stable_config, controller, generator=g_cpu,
            )
            image, x_t, _, _ = wrapper.forward(
                [prompt], latent=x_t, uncond_embeddings=uncond_embeddings,
            )

            segmentor = Segmentor(
                controller, [prompt], args.num_segments,
                args.background_segment_threshold, plot_attn_map=args.plot_attn_map,
            )

            if args.plot_attn_map:
                try:
                    nouns_dict, normalized_nouns_maps = segmentor.attn_per_noun()
                    for idx, noun in nouns_dict:
                        if special_char_pattern.search(noun) or not noun.strip() or "/" in noun:
                            continue
                        map_to_save = normalized_nouns_maps[:, :, idx].astype(np.uint8)

                        if args.plot_attn_mask:
                            threshold = args.background_segment_threshold * 255
                            binary_map = ((map_to_save > threshold) * 255).astype(np.uint8)
                            Image.fromarray(binary_map, mode="L").save(
                                os.path.join(exp_path, f"attnmask_{noun}.jpg")
                            )
                        else:
                            noun_clean = "".join(
                                c for c in noun if c.isalnum() or c in (" ", ".", "_")
                            ).strip()
                            try:
                                Image.fromarray(map_to_save, mode="L").save(
                                    os.path.join(exp_path, f"attention_{noun_clean}.jpg")
                                )
                            except Exception as e:
                                print(f"  Failed to save attention for '{noun_clean}': {e}")
                except Exception as e:
                    print(f"  Error extracting attention maps: {e}")

            del ldm_stable, controller, wrapper, segmentor
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    subfolders = sorted(glob.glob(os.path.join(args.output_dir, "*/")))
    print(f"Total subfolders: {len(subfolders)}")

    for i in range(args.start_index - 1, min(args.end_index, len(subfolders))):
        subfolder_path = subfolders[i]
        print(f"[{i + 1}] {subfolder_path}")
        process_subfolder(args, subfolder_path)
