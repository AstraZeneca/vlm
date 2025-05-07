# Diffusion Instruction Tuning by chen.jin@astrazeneca.com
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor
import abc
import random
import matplotlib as mpl
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import Cache
from .diffag_utils import *
import math


from torchvision.models import resnet50
class ResNetBasedProjection(nn.Module):
    def __init__(self, in_channels=32, out_channels=1, pretrained=True, d_h=256):
        super().__init__()
        # Load ResNet50 (set pretrained to False here, as we will load weights manually)
        self.resnet = resnet50(pretrained=False)
        
        # Modify the first convolution layer to accept in_channels instead of 3 (default for RGB images)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify conv layers in each block to have stride=1 and padding=1 (only for layers with stride != 1)
        self._modify_stride_and_padding(self.resnet.layer1)
        self._modify_stride_and_padding(self.resnet.layer2)
        self._modify_stride_and_padding(self.resnet.layer3)
        self._modify_stride_and_padding(self.resnet.layer4)
        
        # Replace the fully connected layer with a convolutional layer projecting to the desired output
        # This will compress the channel dimension (c) to 1 while keeping the spatial dimensions (w, h)
        self.diffag_norm3 = nn.BatchNorm2d(d_h)
        self.diffag_act = nn.ReLU6(inplace=False)
        self.diffag_conv_last = nn.Conv2d(d_h, out_channels, kernel_size=1, padding=0, stride=1)

    def _modify_stride_and_padding(self, layer):
        """ Modify all conv layers in a block to adjust stride and padding such that spatial dimensions are preserved """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                # Set stride to 1 for all conv layers
                module.stride = (1, 1)
                # Adjust padding based on the kernel size to preserve spatial dimensions
                kernel_size = module.kernel_size[0]  # Assuming square kernels (kernel_size is tuple)
                # Padding calculation: (kernel_size - 1) // 2 to preserve width and height
                ps = (kernel_size - 1) // 2
                module.padding = (ps, ps)
    
    def _normalize_attn(self, attn_mlp: torch.Tensor) -> torch.Tensor:
        """Normalize per-word (dim=-2) attention map."""
        min_vals = attn_mlp.min(dim=-2, keepdim=True)[0]
        max_vals = attn_mlp.max(dim=-2, keepdim=True)[0]
        eps = 1e-6
        return (attn_mlp - min_vals) / (max_vals - min_vals + eps)

    def load_pretrained_weights(self, pretrained_resnet):
        # Load weights for layers that are common between the two models
        # Skip the layers that are changed (conv1 and fc)
        pretrained_dict = pretrained_resnet.state_dict()
        model_dict = self.resnet.state_dict()

        # Filter out layers that don't match (e.g., 'conv1' and 'fc')
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith('conv1') and not k.startswith('fc')
        }

        # Update the model's state_dict with the pretrained weights
        model_dict.update(pretrained_dict)
        # Load the updated state_dict into the model
        self.resnet.load_state_dict(model_dict)

    def forward(self, x):
        # Forward pass through the modified ResNet50 model
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        x = self.diffag_norm3(x)
        x = self.diffag_act(x)
        x = self.diffag_conv_last(x) 
        
        # Ensure the output shape is (b, 1, w, h)
        return x

class DiffagFovProjection(nn.Module):
    def __init__(self, diffag_args, d_in=32, d_h=32, out_channels=1):
        super().__init__()
        self.diffag_args = diffag_args
        print(f'Adding custom conv layers to model')
            
        # Custom convolutional layers for 'fov' configuration
        self.diffag_expand_1 = nn.Conv2d(in_channels=d_in, out_channels=d_h, kernel_size=3, padding=1, bias=False)
        self.diffag_expand_2 = nn.Conv2d(in_channels=d_h, out_channels=d_h, kernel_size=3, padding=1, bias=False)
        self.diffag_squeeze_1 = nn.Conv2d(in_channels=d_h, out_channels=d_in, kernel_size=3, padding=1, bias=False)
        self.diffag_expand_3 = nn.Conv2d(in_channels=d_h, out_channels=d_h, kernel_size=3, padding=1, bias=False)
        self.diffag_expand_4 = nn.Conv2d(in_channels=d_h, out_channels=d_h, kernel_size=3, padding=1, bias=False)

        # BatchNorm or InstanceNorm based on `xatn_norm`
        if self.diffag_args.xatn_norm == 'instance':
            self.diffag_norm1 = nn.InstanceNorm2d(d_h)
            self.diffag_norm2 = nn.InstanceNorm2d(d_h)
            self.diffag_norm3 = nn.InstanceNorm2d(d_in)
        elif self.diffag_args.xatn_norm == 'batch':
            self.diffag_norm1 = nn.BatchNorm2d(d_h)
            self.diffag_norm2 = nn.BatchNorm2d(d_h)
            self.diffag_norm3 = nn.BatchNorm2d(d_in)
        elif self.diffag_args.xatn_norm == 'group':
            self.diffag_norm1 = nn.GroupNorm(d_h, d_h)
            self.diffag_norm2 = nn.GroupNorm(d_h, d_h)
            self.diffag_norm3 = nn.GroupNorm(d_in, d_in)
        elif self.diffag_args.xatn_norm == 'identical':
            # Define identical layers to pass the input unchanged
            self.diffag_norm1 = nn.Identity()
            self.diffag_norm2 = nn.Identity()
            self.diffag_norm3 = nn.Identity()

        # Activation and final projection layers
        if self.diffag_args.xatn_act == 'relu':
            self.diffag_act = nn.ReLU6(inplace=False)
        elif self.diffag_args.xatn_act == 'leakyrelu':    
            self.diffag_act = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.diffag_conv_last = nn.Conv2d(d_in, out_channels, kernel_size=1, padding=0, stride=1)

    def _apply_conv_layers(self, attn_c: torch.Tensor) -> torch.Tensor:
        """Apply deep convolutional layers."""
        return self.diffag_norm3(
            self.diffag_squeeze_1(
                self.diffag_act(
                    self.diffag_expand_4(
                        self.diffag_act(
                            self.diffag_expand_3(
                                self.diffag_act(
                                    self.diffag_norm2(
                                        self.diffag_expand_2(
                                            self.diffag_act(self.diffag_norm1(self.diffag_expand_1(attn_c)))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

    def _apply_simple_conv_layers(self, attn_c: torch.Tensor) -> torch.Tensor:
        """Apply simple convolutional layers."""
        layer1 = self.diffag_act(self.diffag_norm1(self.diffag_expand_1(attn_c)))
        layer2 = self.diffag_act(self.diffag_norm2(self.diffag_expand_2(layer1)))
        return self.diffag_norm3(self.diffag_squeeze_1(layer2))

    def _apply_light_conv_layers(self, attn_c: torch.Tensor) -> torch.Tensor:
        """Apply simple convolutional layers."""
        layer1 = self.diffag_act(self.diffag_norm1(self.diffag_expand_1(attn_c)))
        return self.diffag_norm3(self.diffag_squeeze_1(layer1))

    def _normalize_attn(self, attn_mlp: torch.Tensor) -> torch.Tensor:
        """Normalize per-word (dim=-2) attention map."""
        min_vals = attn_mlp.min(dim=-2, keepdim=True)[0]
        max_vals = attn_mlp.max(dim=-2, keepdim=True)[0]
        eps = 1e-6
        return (attn_mlp - min_vals) / (max_vals - min_vals + eps)

    def forward(self, attn_c):
        # Apply convolution layers based on whether deep conv is enabled
        
        if self.diffag_args.conv_deep == 'light':
            layer3 = self._apply_light_conv_layers(attn_c)
        elif self.diffag_args.conv_deep == 'sim':
            layer3 = self._apply_simple_conv_layers(attn_c)
        elif self.diffag_args.conv_deep == 'deep':
            layer3 = self._apply_conv_layers(attn_c)

        attn_proj = self.diffag_conv_last(self.diffag_act(layer3))
        return attn_proj

def print_grad(grad: torch.Tensor) -> None:
    """
    Print gradient information for debugging purposes.
    
    Args:
        grad (torch.Tensor): The gradient tensor to inspect.
    """
    print(f"Grad-attn_c max: {grad.max()}, grad min {grad.min()}")
    print(f"Grad-attn_c num non-zero grad {(grad != 0).sum()}")
    print(f"Grad-attn_c shape {grad.shape}")

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def extend_mllama_xattn(
    model: nn.Module, 
    controller_f: Callable, 
    diffag_args, 
    eval_mode: bool = False
) -> None:
    """
    Extend MllamaTextCrossAttention layers in the given model with custom attention handling
    to perform cross-attention and store results as needed.

    Args:
        model (nn.Module): The model whose MllamaTextCrossAttention layers need to be modified.
        controller_f (Callable): A function for handling attention processing.
        diffag_args: Configuration arguments for DiffAG.
        eval_mode (bool, optional): If True, enables evaluation mode behavior.
    """
    from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    # Use DummyController if controller_f is None
    if controller_f is None:
        controller_f = DummyController()

    class XattnExtractorMllamaCrossAttention(MllamaTextCrossAttention):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize new projection layers for cross-attention if specified in diffag_args
            if diffag_args.diffag_learn_xattn:
                self.q_proj_diffxattn = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
                self.k_proj_diffxattn = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                self.v_proj_diffxattn = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            
            if diffag_args.diff_trans:
                self.diffxattn_lambda_init = 0.8
                # Use nn.Linear layers with 1 input and output dimension to simulate parameter scalars
                self.diffxattn_lambda_q1 = nn.Linear(self.head_dim, 1, bias=False)
                self.diffxattn_lambda_k1 = nn.Linear(self.head_dim, 1, bias=False)
                self.diffxattn_lambda_q2 = nn.Linear(self.head_dim, 1, bias=False)
                self.diffxattn_lambda_k2 = nn.Linear(self.head_dim, 1, bias=False)

                # Initialize these weights to resemble the intended parameters
                nn.init.normal_(self.diffxattn_lambda_q1.weight, mean=0, std=0.1)
                nn.init.normal_(self.diffxattn_lambda_k1.weight, mean=0, std=0.1)
                nn.init.normal_(self.diffxattn_lambda_q2.weight, mean=0, std=0.1)
                nn.init.normal_(self.diffxattn_lambda_k2.weight, mean=0, std=0.1)
                
        def forward(
            self,
            hidden_states: torch.Tensor,
            cross_attention_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = True,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

            # Perform the standard cross-attention forward pass
            attn_output, attn_weights, past_key_value = super().forward(
                hidden_states=hidden_states,
                cross_attention_states=cross_attention_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                output_attentions=True,  # Ensure attention weights are output for processing
                use_cache=use_cache,
                cache_position=cache_position
            )

            # Determine cross-attention weights based on diffag_args
            if diffag_args.diffag_learn_xattn and cross_attention_states is not None:
                # Apply custom projections for cross-attention
                query_states = self.q_proj_diffxattn(hidden_states)
                key_states = self.k_proj_diffxattn(cross_attention_states)
                # if diffag_args.diff_trans:
                #     # use the same value_states as original xattn
                #     value_states = self.v_proj(cross_attention_states)
                # else:
                value_states = self.v_proj_diffxattn(cross_attention_states)

                # Reshape for multi-head attention
                query_states = query_states.view(hidden_states.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(hidden_states.size(0), -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(hidden_states.size(0), -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                # Repeat `key_states` and `value_states` to match `num_heads`
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # perform norm
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

                # Compute custom cross-attention weights
                cross_attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
                cross_attention_weights = nn.functional.softmax(cross_attention_weights, dim=-1)

                if diffag_args.diff_trans:
                    # following modification from original Differential Transformer to be compitable with fsdp
                    # Create a tensor of ones with the same dimension as head_dim
                    self.diffxattn_lambda_q1.to(query_states.device)
                    self.diffxattn_lambda_k1.to(query_states.device)
                    self.diffxattn_lambda_q2.to(query_states.device)
                    self.diffxattn_lambda_k2.to(query_states.device)
                    input_tensor = torch.ones(self.head_dim, device=query_states.device, dtype=self.diffxattn_lambda_q1.weight.dtype)
                    # Use the forward pass of each layer to simulate the weight multiplication and summation
                    diffxattn_lambda_1 = torch.exp(self.diffxattn_lambda_q1(input_tensor) * self.diffxattn_lambda_k1(input_tensor))
                    diffxattn_lambda_2 = torch.exp(self.diffxattn_lambda_q2(input_tensor) * self.diffxattn_lambda_k2(input_tensor))
                    # Calculate diffxattn_lambda_full without accessing the weights directly
                    diffxattn_lambda_full = diffxattn_lambda_1 - diffxattn_lambda_2 + self.diffxattn_lambda_init

                    cross_attention_weights = attn_weights - diffxattn_lambda_full * cross_attention_weights
                    if diffag_args.diff_trans_output:
                        attn_output_new = torch.matmul(cross_attention_weights, value_states)
                        attn_output_new = attn_output_new.transpose(1, 2).contiguous()
                        attn_output_new = attn_output_new.reshape(attn_output.shape)
                        attn_output_new = self.o_proj(attn_output_new)
                        attn_output = attn_output_new
            else:
                # Use the standard attn_weights if not using custom cross-attention
                cross_attention_weights = attn_weights.clone()

            # # Perform interpolation, reshape, and resize
            n_t_max = diffag_args.max_image_tiles  # Max Number of tiles always = 4, meaning a 2x2 grid is always prepared for each image
            n_t_list = controller_f.num_tiles.cpu()  # Number of tiles vary for each img, mostly = 1, meaning the img only fill (0,0) of the 2x2 grid
            # Check if n_t_list has two dimensions
            if n_t_list.dim() == 2 and n_t_list.size(0) == 1:
                # Remove the outer layer
                n_t_list = n_t_list.squeeze(0)
            grid_mask = (controller_f.cross_attention_mask_raw.max(1))[0].squeeze(1) # binary (0,1) mask of shape b n_t_max indiciting where tiles are filled (1) and not filled (0)
            # in the 2x2 grid, grid_mask[b] = [1,0,0,0] means top-left, [0,1,0,0] means top-right, [0,0,1,0] means bottom-left and [0,0,0,1] means bottom-right

            # Initial shape of cross_attention_weights: [b, h, n1, n2], where n2 = n_t * n_p
            # n1 = len(text emb), n2 = num of patch tokens, h = num of heads
            b, h, n1, n2 = cross_attention_weights.shape

            attn_canvas_list = []  # Store each processed batch element's attention canvas
            grid_size = int(np.sqrt(n_t_max))  # Target grid size (2x2 for n_t_max=4)
            # loop over each instance in the batch to handle potential different number of n_t
            for batch_idx in range(b):
                n_t = int(n_t_list[batch_idx])  # Actual number of tiles for this instance
                grid_mask_b = grid_mask[batch_idx]  # Binary mask indicating tile positions for the current instance

                # Step 1: Separate the patches into tiles for `n_t_max`
                n_p = n2 // n_t_max  # Number of patch tokens per tile
                attn_c = rearrange(
                    cross_attention_weights[batch_idx], "h n1 (n_t n_p) -> n_t h n1 n_p", n_t=n_t_max, n_p=n_p
                )

                # Step 2: Interpolate each tile’s patches to be approximately square
                tile_res = int(np.ceil(np.sqrt(n_p)))  # Target size to approximate `n_p` as a square
                attn_c = rearrange(attn_c, "n_t h n1 n_p -> (n1 n_t) h n_p")  # Flatten n1 and tiles for interpolation
                attn_c = F.interpolate(attn_c, size=tile_res * tile_res, mode="linear", align_corners=False)
                attn_c = rearrange(
                    attn_c, "(n1 n_t) h (r1 r2) -> n1 h n_t r1 r2", r1=tile_res, r2=tile_res, n1=n1, n_t=n_t_max
                )

                # Step 3: Initialize an appropriate canvas based on `n_t` and `grid_mask`
                if n_t == 1:
                    # Only one tile, use it directly without a canvas
                    attn_canvas = attn_c[:, :, 0]
                else:
                    # Determine the canvas size based on `grid_mask`
                    if n_t == 2:
                        if grid_mask_b[0] == 1 and grid_mask_b[1] == 1:  # Horizontal arrangement
                            canvas_res = (tile_res, tile_res * 2)  # height, width
                        elif grid_mask_b[1] == 1 and grid_mask_b[2] == 1:  # Vertical arrangement
                            canvas_res = (tile_res * 2, tile_res)
                        else:
                            raise ValueError("Invalid grid_mask configuration for n_t=2.")
                    else:
                        canvas_res = (tile_res * grid_size, tile_res * grid_size)  # 2x2 arrangement for `n_t=4`

                    # Initialize the canvas with the computed resolution
                    attn_canvas = torch.zeros(n1, h, *canvas_res, device=attn_c.device)

                    # Step 3-1: Place each tile according to `grid_mask`
                    for idx in range(n_t_max):
                        if grid_mask_b[idx] == 1:
                            # Calculate tile positions for 2x2 grid (top-left, top-right, bottom-left, bottom-right)
                            row = idx // grid_size
                            col = idx % grid_size
                            r_start, r_end = row * tile_res, (row + 1) * tile_res
                            c_start, c_end = col * tile_res, (col + 1) * tile_res
                            attn_canvas[:, :, r_start:r_end, c_start:c_end] = attn_c[:, :, idx]

                # Step 4: Resize to final output resolution (32x32) which is the size SD attention maps been prepared
                resize = Resize((32, 32))
                attn_canvas_resized = resize(attn_canvas) # n1 h 32 32

                # Step 5: Transform to match intended output shape and store in the list
                attn_proj = rearrange(attn_canvas_resized, "n1 h r1 r2 -> h (r1 r2) n1")
                attn_canvas_list.append(attn_proj)

            # Combine all processed instances back into a batch
            attn_proj_batch = torch.stack(attn_canvas_list) # b h (r1 r2) n1

            # Controller processing
            if eval_mode or attn_proj_batch.size(0) == diffag_args.batch_size_eval:
                attn_proj_batch = controller_f(attn_proj_batch.clone(), True, "mid")

            # Return standard outputs
            return attn_output, attn_weights, past_key_value

    # Function to replace MllamaTextCrossAttention layers
    def replace_attention_layer(net: nn.Module, count: int = 0) -> int:
        for name, module in net.named_children():
            if (
                isinstance(module, MllamaTextCrossAttention)
                and hasattr(module, 'layer_idx')
                and (len(diffag_args.diff_attend_layers) != 0 and module.layer_idx in diffag_args.diff_attend_layers)
            ):
                print(f"Replacing {name}, layer={module.layer_idx} with XattnExtractorMllamaCrossAttention")
                setattr(net, name, XattnExtractorMllamaCrossAttention(module.config, layer_idx=module.layer_idx))
                count += 1
            elif (
                isinstance(module, MllamaTextCrossAttention)
                and hasattr(module, 'layer_idx')
                and len(diffag_args.diff_attend_layers) == 0
                and (module.layer_idx + 1) % diffag_args.diff_attend_layer_every == 0
            ):
                print(f"Replacing {name}, layer={module.layer_idx} with XattnExtractorMllamaCrossAttention")
                setattr(net, name, XattnExtractorMllamaCrossAttention(module.config, layer_idx=module.layer_idx))
                count += 1
            else:
                count = replace_attention_layer(module, count)
        return count

    # Execute replacement and count replaced layers
    controller_f.num_att_layers = replace_attention_layer(model)
    print(f"Total MllamaTextCrossAttention layers replaced: {controller_f.num_att_layers}")


def extend_llama_attention_with_xattn(
    model: nn.Module, 
    controller_f: Callable, 
    diffag_args, 
    eval_mode: bool = False
    ) -> None:
    """
    Extend LlamaAttention layers in the given model with XattnExtractorLlamaAttention,
    extending their forward method to handle cross-attention extraction and storage.
    Prints the total number of replacements made.

    Args:
        model (nn.Module): The model whose LlamaAttention layers need to be replaced.
    """

    class XattnExtractorLlamaAttention(LlamaAttention):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if diffag_args.diffag_learn_xattn:
                # Initialize new projection layers for cross-attention
                self.q_proj_diffxattn = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
                self.k_proj_diffxattn = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
                self.v_proj_diffxattn = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value: Optional[Cache] = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    cache_position: Optional[torch.LongTensor] = None,
                    **kwargs,  # Catch all other keyword arguments
                    ):
            # Extract vision_mask and text_mask if provided in kwargs
            # vision_mask = kwargs.get('vision_mask', None)
            # text_mask = kwargs.get('text_mask', None)
            if controller_f.current_inputs is not None:
                text_mask, vision_mask = extract_masks(controller_f.current_inputs, hidden_states)
            else:
                text_mask, vision_mask = None, None

            # Forward pass for standard attention
            if diffag_args.causal_mirror:
                return_causal_mask = True
            else:
                return_causal_mask = False

            attn_returns = super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=True,  # Ensure attention weights are output
                    use_cache=use_cache,
                    cache_position=cache_position,
                    causal_unmask=diffag_args.causal_unmask,
                    return_causal_mask=return_causal_mask,
                    diffag_mean_lora_atn=diffag_args.diffag_mean_lora_atn,
                    diffag_mean_lora_scale=diffag_args.diffag_mean_lora_scale,
                )

            if diffag_args.causal_mirror and not diffag_args.diffag_learn_xattn:
                if diffag_args.diffag_mean_lora_atn:
                    causal_mask_return, value_states, attn_weights, present_key_value, attn_weights_avg = attn_returns
                else:
                    causal_mask_return, value_states, attn_weights, present_key_value = attn_returns
            else:
                if diffag_args.diffag_mean_lora_atn:
                    attn_output, attn_weights, present_key_value, attn_weights_avg = attn_returns
                else:
                    attn_output, attn_weights, present_key_value = attn_returns

            # Extract cross-attention weights if vision_mask and text_mask are provided
            if vision_mask is not None and text_mask is not None:
                # Initialize an empty list to collect the resized cross-attention weights for each batch
                all_attn_c = []
                # Determine the maximum n1 across the batch
                max_n1 = max(text_mask.sum(dim=-1)).item()

                # compute cross-attention using new weights
                if diffag_args.diffag_learn_xattn:
                    # Loop over each batch
                    for batch_idx in range(hidden_states.size(0)):  # Loop over batch dimension (b)
                        # Extract the text and vision tokens based on the mask for the current batch
                        text_indices = text_mask[batch_idx].nonzero(as_tuple=True)[0]
                        vision_indices = vision_mask[batch_idx].nonzero(as_tuple=True)[0]

                        # Gather the tokens from hidden_states for the current batch
                        text_tokens = hidden_states[batch_idx, text_indices, :]
                        vision_tokens = hidden_states[batch_idx, vision_indices, :]

                        # Use new projection layers for cross-attention
                        query_states = self.q_proj_diffxattn(text_tokens)  # New q_proj for cross-attention
                        key_states = self.k_proj_diffxattn(vision_tokens)  # New k_proj for cross-attention
                        value_states = self.v_proj_diffxattn(vision_tokens)  # New v_proj for cross-attention

                        # Reshape for multi-head attention (with batch size of 1 to handle n1 and n2 separately)
                        query_states = query_states.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # shape: (h, n1, d)
                        key_states = key_states.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # shape: (h, n2, d)
                        # value_states = value_states.view(value_states.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

                        # Compute cross-attention between text queries and vision keys
                        cross_attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
                        cross_attention_weights = nn.functional.softmax(cross_attention_weights, dim=-1)  # Apply softmax to normalize

                        # Handle varying number of tokens in each batch by padding to max_n1
                        # Reshape for further processing: n1 = len(text emb), n2 = num of patch tokens, h = num of heads
                        attn_c = rearrange(cross_attention_weights.clone(), "h n1 n2 -> n1 n2 h")

                        n1, n2, h = attn_c.shape
                        res = int(np.ceil(np.sqrt(n2)))  # Get the smallest integer r such that r*r >= n2

                        # Interpolate the attention from (n1, n2, h) to (n1, res*res, h)
                        attn_c = F.interpolate(attn_c.permute(2, 0, 1), size=res * res, mode='linear', align_corners=False)
                        attn_c = attn_c.permute(1, 2, 0)  # Rearrange back to (n1, res*res, h)

                        # Reshape and resize: r1 x r2 = res * res, where n2 is the number of patch tokens
                        attn_c = rearrange(attn_c, "n1 (r1 r2) h -> n1 h r1 r2", r1=res, r2=res)
                        resize = Resize((32, 32))  # Assuming you want to resize to 32x32
                        attn_c = resize(attn_c)

                        # Pad n1 to max_n1 by adding zeros to the right
                        padding = (0, 0, 0, 0, 0, 0, 0, max_n1 - n1)  # Padding only the n1 dimension on the right
                        attn_c = F.pad(attn_c, padding, value=0)

                        # Reshape back to 1 n1 h r1 r2 format
                        attn_c = rearrange(attn_c, "n1 h r1 r2 -> 1 n1 h r1 r2", n1=max_n1, h=h)

                        # Append the processed attention map for the current batch to the list
                        all_attn_c.append(attn_c)

                # collect cross-attention from self-attention with pretrained weights
                else:
                    # Loop over each batch
                    if diffag_args.diffag_mean_lora_atn:
                        attn_weights_c = attn_weights_avg.clone()
                    else:
                        attn_weights_c = attn_weights.clone()
                        
                    for i in range(attn_weights_c.size(0)):  # Loop over batch dimension (b)
                        
                        # Extract indices of True values in text_mask and vision_mask for this batch
                        text_indices = text_mask[i].nonzero(as_tuple=True)[0]
                        vision_indices = vision_mask[i].nonzero(as_tuple=True)[0]

                        # Gather the relevant slices from attn_weights_c for this batch
                        cross_attention_weights = attn_weights_c[
                            i,  # Select the i-th batch
                            :,  # Keep all heads
                            text_indices,  # Select relevant text indices for the first N dimension
                        ][:, :, vision_indices]  # Select relevant vision indices for the second N dimension
                        if (
                            diffag_args.causal_mirror 
                            and not (
                                diffag_args.diffag_pretrain_diffag_net != 0
                                and diffag_args.mirror_detach_staged
                                and controller_f.global_step >= diffag_args.diffag_pretrain_diffag_net
                            )
                        ):
                            # block gradients passing to the original causal attention weights
                            cross_attention_weights = cross_attention_weights.detach()

                        if diffag_args.causal_unmask:
                            cross_attention_weights_mirror = attn_weights_c[
                                i,  # Select the i-th batch
                                :,  # Keep all heads
                                vision_indices,  # Select relevant vision indices for the first N dimension
                            ][:, :, text_indices]  # Select relevant text indices for the second N dimension
                            # Transpose the last two dimensions of cross_attention_weights_mirror
                            cross_attention_weights_mirror = cross_attention_weights_mirror.transpose(-1, -2)
                            # Adding the mirrored cross-attention weights as learnable residual
                            cross_attention_weights = cross_attention_weights + cross_attention_weights_mirror
                            if diffag_args.causal_mirror:
                                # Convert causal_mask_return from -inf/0 to 0/1
                                # Use torch.where to create the binary mask (1 for 0, 0 for -inf)
                                binary_mask = torch.where(
                                    causal_mask_return == 0, 
                                    torch.tensor(1.0, device=causal_mask_return.device), 
                                    torch.tensor(0.0, device=causal_mask_return.device)
                                    ).to(attn_weights.dtype)
                                # Apply the mask by multiplying with attn_weights
                                attn_weights = attn_weights * binary_mask
                                if diffag_args.diffag_mean_lora_atn:
                                    attn_weights_avg = attn_weights_avg * binary_mask
                    
                        # cross_attention_weights.shape = h n1 n2
                        # Reshape for further processing: n1 = len(text emb), n2 = num of patch tokens, h = num of heads
                        attn_c = rearrange(cross_attention_weights.clone(), "h n1 n2 -> n1 n2 h")
                        
                        n1, n2, h = attn_c.shape
                        # Calculate the target size for interpolation (assuming square target size)
                        res = int(np.ceil(np.sqrt(n2)))  # Get the smallest integer r such that r*r >= n2

                        # Interpolate attn_c from (n1, n2, h) to (n1, res*res, h)
                        attn_c = F.interpolate(attn_c.permute(2, 0, 1), size=res * res, mode='linear', align_corners=False)
                        attn_c = attn_c.permute(1, 2, 0)  # Rearrange back to (n1, res*res, h)

                        # Reshape and resize: r1 x r2 = res * res, where n2 is the number of patch tokens
                        attn_c = rearrange(attn_c, "n1 (r1 r2) h -> n1 h r1 r2", r1=res, r2=res)
                        resize = Resize((32, 32))  # Assuming you want to resize to 32x32
                        attn_c = resize(attn_c)

                        # Pad n1 to max_n1 by adding zeros to the right
                        padding = (0, 0, 0, 0, 0, 0, 0, max_n1 - n1)  # Padding only the n1 dimension on the right
                        attn_c = F.pad(attn_c, padding, value=0)

                        # Reshape back to 1 n1 h r1 r2 format
                        attn_c = rearrange(attn_c, "n1 h r1 r2 -> 1 n1 h r1 r2", n1=max_n1, h=h)

                        # Append to the list, note that it now has a batch dimension of 1
                        all_attn_c.append(attn_c)

                # Concatenate all the resized tensors to form the final batch tensor
                # The final shape will be [b, (n1 * h), 32, 32] across all batches
                final_attn_c = torch.cat(all_attn_c, dim=0)

                # Third, apply conv layers and pass attention to store
                attn_proj = rearrange(final_attn_c, "b n1 h r1 r2 -> (b n1) h r1 r2", n1=max_n1, h=h)
                # layer3 = self._apply_conv_layers(attn_c) if diffag_args.conv_deep else self._apply_simple_conv_layers(attn_c)
                # attn_proj = self.diffag_conv_last(self.diffag_act(layer3))  # e.g. (b n1) 1 32 32
                # if not torch.isnan(attn_proj).all():
                attn_mlp = rearrange(attn_proj, "(b n1) h r1 r2 -> b h (r1 r2) n1", n1=max_n1)
                # attn_mlp = self._normalize_attn(attn_mlp)
                is_cross = True
                if eval_mode:
                    attn_mlp = controller_f(attn_mlp.clone(), is_cross, "mid")
                else:
                    if attn_mlp.shape[0] == diffag_args.batch_size_eval:
                        attn_mlp = controller_f(attn_mlp.clone(), is_cross, "mid")
                # else:
                #     print("attn_proj is all nan due to media_locations mask, skip attention_store")
                
            if diffag_args.causal_mirror and not diffag_args.diffag_learn_xattn:
                if diffag_args.diffag_mean_lora_atn:
                    return super().forward(
                        hidden_states=hidden_states,
                        cache_weights_key=[value_states, attn_weights, present_key_value, attn_weights_avg],
                        diffag_mean_lora_scale=diffag_args.diffag_mean_lora_scale,
                    )
                else:
                    return super().forward(
                        hidden_states=hidden_states,
                        cache_weights_key=[value_states, attn_weights, present_key_value],
                        diffag_mean_lora_scale=diffag_args.diffag_mean_lora_scale,
                    )
            else:
                # Return regular attention outputs if masks are not provided
                return attn_output, attn_weights, present_key_value


    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller_f is None:
        controller_f = DummyController()

    def replace_attention_layer(net: nn.Module, count: int = 0) -> int:
        """
        Recursively replace all instances of LlamaAttention with XattnExtractorLlamaAttention
        and count the number of replacements.

        Args:
            net (nn.Module): The network in which to replace the attention layers.
            count (int): Initial count of replaced layers.

        Returns:
            int: Total count of replaced LlamaAttention layers.
        """
        for name, module in net.named_children():
            if (
                module.__class__.__name__ == "LlamaAttention"
                and hasattr(module, 'layer_idx')
                and len(controller_f.diffag_args.diff_attend_layers) != 0
                and (module.layer_idx) in controller_f.diffag_args.diff_attend_layers
            ):
                print(f"Replacing {name}, layer={module.layer_idx} with XattnExtractorLlamaAttention")
                setattr(net, name, XattnExtractorLlamaAttention(config=module.config, layer_idx=module.layer_idx))
                count += 1
            elif (
                module.__class__.__name__ == "LlamaAttention"
                and hasattr(module, 'layer_idx')
                and len(controller_f.diffag_args.diff_attend_layers) == 0
                and (module.layer_idx + 1) % controller_f.diffag_args.diff_attend_layer_every == 0
            ):
                print(f"Replacing {name}, layer={module.layer_idx} with XattnExtractorLlamaAttention")
                setattr(net, name, XattnExtractorLlamaAttention(config=module.config, layer_idx=module.layer_idx))
                count += 1
            else:
                count = replace_attention_layer(module, count)
        return count

    controller_f.num_att_layers = replace_attention_layer(model)
    print(f"Total LlamaAttention layers replaced: {controller_f.num_att_layers} on device={model.device}")

class AttentionStore(abc.ABC):
    """
    A class for managing and storing attention weights during the prediction process 
    in a language model (LLM), particularly focusing on accumulating cross-attention 
    maps across different layers for each step of next-word prediction.
    """

    def __init__(self, average_att_time: bool = True, store_last: bool = True, diffag_args=None):
        """
        Initialize the AttentionStore.

        Args:
            average_att_time (bool): Whether to average attention over time.
            store_last (bool): Whether to store only the last cross-attention layer.
        """
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.average_att_time = average_att_time
        self.store_last = store_last
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.answering = False
        self.current_inputs = None
        self.diffag_args = diffag_args

    @staticmethod
    def get_empty_store():
        """Return an empty store dictionary for storing attention data."""
        return {"mid_cross": []}

    def step_callback(self, x_t):
        """Callback for processing the input tensor during each step."""
        return x_t

    def between_steps(self):
        """
        Handle the processing of attention data between steps. 
        Accumulates attention maps across layers after each prediction step.
        """
        if not self.attention_store:
            self.attention_store = self.step_store
            self.answering = True
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    if isinstance(self.step_store[key][i], list):
                        for b in range(len(self.step_store[key][i])):
                            self.attention_store[key][i][b] = torch.cat(
                                (
                                    self.attention_store[key][i][b],
                                    self.step_store[key][i][b].clone(),
                                ),
                                dim=-1,
                            )
                    else:
                        self.attention_store[key][i] = torch.cat(
                            (
                                self.attention_store[key][i],
                                self.step_store[key][i].clone(),
                            ),
                            dim=-1,
                        )

        self.step_store = self.get_empty_store()

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        Process and store attention weights during a prediction step.

        Args:
            attn (torch.Tensor): The attention weights tensor.
            is_cross (bool): Whether this is cross-attention.
            place_in_unet (str): The place in the U-Net model (e.g., "mid").
        
        Returns:
            torch.Tensor: The processed attention weights.
        """
        if place_in_unet != "mid":
            return attn
        
        if attn.shape[1] <= 32**2:  # Avoid memory overhead for large attention maps
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn.shape[0] > 1:
                self.step_store[key].append([attn[b] for b in range(attn.shape[0])])
            else:
                self.step_store[key].append(attn[0])
        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """
        Process the attention weights for each layer in a prediction step.

        This method accumulates cross-attention maps across layers, and once all layers
        for a step are processed, it aggregates them for the current prediction step.

        Args:
            attn (torch.Tensor): The attention weights tensor.
            is_cross (bool): Whether this is cross-attention.
            place_in_unet (str): Placeholder key corresponds to the place in the U-Net model (e.g., "mid").
        
        Returns:
            torch.Tensor: The processed attention weights.
        """
        
        self.cur_att_layer += 1
        attn[:] = self.forward(attn[:], is_cross, place_in_unet)

        if self.cur_att_layer == self.num_att_layers and place_in_unet == "mid":
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return attn

    def get_average_attention(self, average_att_time: bool = True):
        """
        Compute and return the average attention maps across time steps.

        Args:
            average_att_time (bool): Whether to average the attention over time.

        Returns:
            dict: A dictionary with averaged attention maps.
        """
        self.average_att_time = average_att_time
        return {
            key: [item for item in self.attention_store[key]]
            for key in self.attention_store
        }

    def reset(self):
        """Reset the AttentionStore to its initial state."""
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def compute_cross_attention_flow(
    layer_attention_store, 
    sum_propogate=False, 
    diffag_reg_propogate=False, 
    min_max_aggregation=False
    ):
    """
    Compute the cross attention flow for aggregating attention maps through all transformer heads and layers.
    
    Args:
        layer_attention_store (torch.Tensor): Attention from all layers, shape (L, B, H, r1r2, n1).

    Returns:
        torch.Tensor: Aggregated attention map of shape (B, r1r2, n1).
    """
    
    # Initialize variables
    L, B, H, r1r2, n1 = layer_attention_store.shape  # (num_layers, batch, heads, r1r2, n1)
    
    # Step 1: Head aggregation - Aggregating heads for each layer
    if min_max_aggregation:
        # Compute max queries (dimension corresponding to text tokens) and mean over over keys (dimension corresponding to image tokens)
        # our intuiation is for each head, find the most attended word (text toekn), and estimate its weight by average of image tokens
        W = torch.mean(torch.max(layer_attention_store, dim=-1).values, dim=-1)  # Shape (L, B, H)
        
        # Reshape W to be broadcastable across S
        W = W.unsqueeze(-1).unsqueeze(-1)  # Shape (L, B, H, 1, 1)
        
        # Step 2: Apply head weights
        S_prime = layer_attention_store * W  # Shape (L, B, H, r1r2, n1)

        # Step 3: average heads for each layer
        S_prime = torch.mean(S_prime, dim=2)  # Shape (L, B, r1r2, n1)
    else:
        # L, B, H, r1r2, n1 
        S_prime = layer_attention_store  # Keep per-head attention

    # Initialize the final attention map to aggregate over layers
    aggregated_attention_map = S_prime[0]  # Start with the first layer, shape (B, H, r1r2, n1)

    # Propagate attention through layers using pointwise multiplication
    for l in range(1, L):
        if sum_propogate:
            aggregated_attention_map = aggregated_attention_map + S_prime[l]  # Sum attention across layers
        else:
            aggregated_attention_map = aggregated_attention_map * (S_prime[l])  # Element-wise multiplication to aggregate attention across layers
    
    # Step 4: Apply regularization along the text token dimension (n1)
    if diffag_reg_propogate:
        with torch.no_grad():  # Ensure no gradients are computed during the regularization step
            # Compute the unmasked length L0 (simulating it as the full length of text tokens, n1)
            unmasked_len = n1

            # Compute the regularization term: 1 - (L0 - 1) / L
            # constrain impact of early tokens, weights later tokens more
            regularization_term = (torch.arange(unmasked_len, device=aggregated_attention_map.device) + 1) / unmasked_len
            if min_max_aggregation:
                regularization_term = regularization_term.unsqueeze(0).unsqueeze(0).expand(B, r1r2, -1)  # Shape (B, r1r2, n1)
            else:
                regularization_term = regularization_term.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, H, r1r2, -1)  # Shape (B, H, r1r2, n1)

        # Apply the regularization term to the aggregated attention map without affecting gradients
        aggregated_attention_map_reg = aggregated_attention_map * regularization_term.detach()
    else:
        aggregated_attention_map_reg = aggregated_attention_map
    # aggregated_attention_map_reg, _ = torch.max(aggregated_attention_map_reg, dim=1)  # Shape (B, r1r2, n1)
    # Normalize final attention (softmax over the dimension of image tokens r1r2, i.e. the sum of all image toekens equals to one)
    # aggregated_attention_map_sfotmax = nn.functional.softmax(aggregated_attention_map_reg * 1e30, dim=-2)
    
    # Return the final aggregated attention map
    return aggregated_attention_map  # Final shape (B, r1r2, n1) if min_max_aggregation else (B, H, r1r2, n1)

def aggregate_attention(
    attention_store: AttentionStore,
    from_where: List[str] = ["mid"],
    is_cross: bool = True,
    average_att_ch: bool = True,
    average_att_time: bool = True,
    use_layer: str = "last",
    num_last_layers: int = 1,
    attn_select_layer: list = [0, -1],
) -> torch.Tensor:
    """
    Aggregates attention maps from different layers or heads.

    Args:
        attention_store (AttentionStore): Object storing attention maps.
        res (int): Resolution of attention maps.
        from_where (List[str]): List specifying which layers to extract attention from.
        is_cross (bool): Whether to aggregate cross-attention or self-attention.
        average_att_ch (bool): Whether to average attention across channels.
        average_att_time (bool): Whether to average attention across time steps.
        use_layer (str): Strategy for selecting layers ("last", "select", or "show_all").
        num_last_layers (int): Number of last layers to aggregate attention from.
        attn_select_layer (list): List of specific layers to select attention from.

    Returns:
        torch.Tensor: Aggregated attention map.
    """
    out = []
    attention_maps = attention_store.get_average_attention(average_att_time=average_att_time)
    num_layers = len(attention_maps[f"{from_where[0]}_{'cross' if is_cross else 'self'}"])
    
    if isinstance(attention_maps["mid_cross"][0], list):
        res = int(np.sqrt(attention_maps["mid_cross"][0][0].shape[-2]))
    else:
        res = int(np.sqrt(attention_maps["mid_cross"][0].shape[-2]))
    
    num_pixels = res ** 2

    for location in from_where:
        for i, item in enumerate(attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]):
            if use_layer == "last" and i < (num_layers - num_last_layers):
                continue
            elif use_layer == "select" and i not in attn_select_layer:
                continue
            # if isinstance(item, list):
            #     cross_maps = item[select] if item[0].shape[-2] == num_pixels else None
            # else:
            #     cross_maps = item if item.shape[-2] == num_pixels else None
            # if cross_maps is not None:
            # out.append(item)
            
            # items is a list of two tensors h (r1 r2) n1
            if isinstance(item, list):
                stacked_item = torch.stack(item, dim=0)  # Stack tensors in each item along a new dimension
            else:
                stacked_item = torch.stack([item], dim=0)  # Stack tensors in each item along a new dimension
            # Now stacked_item has shape b h (r1 r2) n1
            out.append(stacked_item)  # Append stacked_item to out

    # if use_layer == "show_all":
    #     for i in range(len(out)):
    #         out[i] = out[i].sum(0) / out[i].shape[0] if average_att_ch else out[i].max(0)[0]
    #     return out
    # else:
    #     out = torch.cat(out, dim=0)
    #     return out.sum(0) / out.shape[0] if average_att_ch else out.max(0)[0]

    # After appending all stacked items, you can compute the mean
    if len(out) > 0:
        stacked_out = torch.stack(out, dim=0)  # Stack all items in out
        # mean_tensor = stacked_out.mean(dim=0)  # Compute the mean along the first dimension
    return stacked_out


def compute_perword_attention_dict(
    attention_map: torch.Tensor,
    tokens: str,
) -> Dict[str, torch.Tensor]:
    """
    Computes and returns a dictionary mapping words to their corresponding attention maps.

    Args:
        attention_map (torch.Tensor): projected attention map (ch=1) of single instance.
        prompts (list): List of prompts for which attention maps are computed.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping words to their corresponding attention map tensors.
    """

    res = int(np.sqrt(attention_map.shape[0]))
    attention_map = attention_map.reshape(res, res, -1)
    
    attn_dict = {}
    accumulated_attention = {}
    word_counts = {}
    vis_len = min(len(tokens), attention_map.shape[-1])

    # Accumulate attention maps for each word
    for i in range(vis_len):
        word = tokens[i].strip()
        if word not in accumulated_attention:
            accumulated_attention[word] = attention_map[:, :, i].clone()
            word_counts[word] = 1
        else:
            accumulated_attention[word] += attention_map[:, :, i]
            word_counts[word] += 1

    # Process each unique word
    for word, attention_sum in accumulated_attention.items():
        # Compute the average attention map
        average_attention = attention_sum / word_counts[word]
        # Normalize and resize the attention map
        min_val = average_attention.min()
        max_val = average_attention.max()
        range_val = max_val - min_val + 1e-6  # Adding epsilon to avoid division by zero
        average_attention = (average_attention - min_val) / range_val

        resize = Resize((32, 32))
        attn_dict[word] = resize(average_attention.unsqueeze(0)).squeeze(0)
    
    return attn_dict