# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .embeddings import Timesteps, TimestepEmbedding
from .activations import get_activation


class NoiseAdapter(nn.Module):
    r"""
    The `TimeAwareEncoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
        resnet_time_scale_shift (`str`, defaults to `"default"`)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (64,),
        resnet_time_scale_shift: str = "scale_shift",
        freq_shift: int = 0,
        non_linearity: str = "swish",
        flip_sin_to_cos: bool = True,
    ):
        super().__init__()

        timestep_input_dim = max(128, block_out_channels[0])
        self.time_proj = Timesteps(timestep_input_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_input_dim, block_out_channels[0] * 2)
        
        self.nonlinearity = get_activation(non_linearity)
        self.conv1 = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        self.gradient_checkpointing = False

    def forward(
            self,
            sample: torch.Tensor,
            timesteps: int,
            scale: float = 1.0
            ) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        # time embedding
        bsz = sample.size(0)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=list(self.time_embedding.parameters())[0].dtype)
        emb = self.time_embedding(t_emb).view(bsz, -1, 1, 1)

        hidden_states = self.conv1(sample)
        hidden_states = self.nonlinearity(hidden_states)
        scale, shift = torch.chunk(emb, 2, dim=1)
        # print(torch.mean(scale))
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.conv2(hidden_states)

        return hidden_states

