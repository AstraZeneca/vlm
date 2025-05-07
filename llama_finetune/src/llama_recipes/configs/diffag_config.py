# Diffusion Instruction Tuning by chen.jin@astrazeneca.com
# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import transformers
from torchvision.models import resnet50
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Tuple

@dataclass
class diffag_config:
    diff_xattn_guidence: bool = False  # Diffusion X-Attn Guidance
    conv_deep: str = "sim"  # choose from light, sim, deep
    batch_size_eval: int = 1  # Batch size during evaluation
    diff_attend_layer_every: int = 1  # Extract attention every number of layers defined here
    layer_extract: str = "average"  # average or max
    use_layer: str = "average_all"  # average_all=average of all attn layers, last=use layers after num_last_layers, select=use attn_select_layer
    num_last_layers: int = 1  # Default of 1 counts only the last layer
    attn_select_layer: List[int] = field(default_factory=lambda: [0, 7])  # Specify xattn layer to aggregate
    plot_step: int = 10  # Save attn plot every plot_step
    sd_xattn_loss_scale: float = 10.0  # Scale sd_xattn_loss to same scale as original
    mse_decay: bool = False  # apply mse weight decay start with 1 and decrease to 0
    mse_decay_factor: float = 1.0  # Slow: 1/25, 1/5; Medium: 1; Fast: 5, 25
    inverse_mse_decay: bool = False  # When inverse, mse weight starts at 0 and increases to 1
    image_train_attn_path: str = ""  # Set diffag data path to trigger diffag dataload
    max_n1: int = 30  # max_n1 is max number of text token to exclude sparse attention
    noun_only: bool = False  # strict loss strategy only compute diffag loss for noun
    adj_only: bool = False  # strict loss strategy only compute diffag loss for adjective
    is_nav: bool = False  # strict loss strategy only compute diffag loss for noun, adjective, and verb
    sub_obj: bool = False  # strict loss strategy only compute diffag loss for subject and object
    xatn_norm: str = "instance"  # normalization to be used in diff attn guidance layers, choose from [instance, batch]
    diff_guidance_net: str = "fov"  # default fov is 3-6 custom conv layers, options: [resnet50pretrain, resnet50scratch]
    diffag_hidden_dim: int = 32  # hidden dimension of diffag nets
    diffag_pretrain_diffag_net: int = 0  # optional pretrain the diff_guidance_net
    diffag_pretrain_lr_scale: float = 1.0  # optional scale learning rate during pretrain the diff_guidance_net
    causal_unmask: bool = False  # unmask the causal mask, attending future tokens
    causal_mirror: bool = False  # use mirror causal mask for diffag loss
    len_reg_mse: bool = True  # use len_loss_mse to regularize average_loss_attn
    mirror_detach_staged: bool = False  # apply detach logic during the 1st stage
    sub_obj_exclude_reject: bool = False  # strict loss strategy to extract_before_rejected_answer
    loss_linear_alpha: bool = False  # if true in stage 2 linear scale loss
    diff_attend_layers: str = ""  # Specify xattn layer to aggregate
    diffag_mean_layer: bool = False  # whether to average layer_attention_store before diffag_projection
    diffag_mean_lora_atn: bool = False  # reverse compute attn_weights before lora and return mean
    diffag_lora_r: int = 64  # lora_r and lora_alpha
    diffag_mean_lora_scale: float = 1.0  # scaling factor for attn_weights_base (before lora)
    diffag_learn_xattn: bool = False  # initialize new layers to learn cross-attention instead of extracting from self-attn
    diffag_xattn_flow: bool = False  # Compute cross attention flow for aggregating attention maps
    diffag_sum_propogate: bool = False  # default: elementwise multiplication
    diffag_reg_propogate: bool = False  # regularize along text token dim, i.e., weight later tokens more
    diffag_plot_before_proj: bool = False  # for debugging, plot attn_c before conv projections
    diffag_skip_proj: bool = False  # optional skip conv projections
    diffag_maxmax: bool = False  # apply max-max attention aggregation, max over layer and head dimensions
    diffag_meanmean: bool = False  # apply mean-mean attention aggregation, max over layer and head dimensions
    xatn_act: str = "relu"  # default: relu; for xattn_flow, use leakyrelu
    max_image_tiles: int = 4  # Default of 4 for mllama 3.2
    diff_trans: bool = False  # Differential Transformer alike noise cancellation
    diff_trans_output: bool = False  # return attention output with the differential xattention
