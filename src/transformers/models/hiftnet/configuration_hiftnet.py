# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class HiFTNetConfig(PretrainedConfig):
    # TODO: @eustlb, check this
    model_type = "generator"
    base_config_key = "generator_config"

    def __init__(
        self,
        hidden_size=512,
        num_mel_bins=80,
        n_fft=16,
        hop_length=5,
        sampling_rate=24000,
        num_layers=3,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        nsf_kernel_sizes=[30, 6, 1],
        nsf_strides=[15, 3, 1],
        nsf_padding=[8, 2, 0],
        nsf_residual_block_kernel_sizes=[7, 7, 11],
        mrf_kernel_sizes=[3, 7, 11],
        mrf_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        **kwargs,
    ):
        
        infered_num_layers = set(
            [
                len(upsample_rates),
                len(upsample_kernel_sizes),
                len(nsf_kernel_sizes),
                len(nsf_strides),
                len(nsf_padding),
                len(nsf_residual_block_kernel_sizes),
            ]
        )

        if len(infered_num_layers) != 1 or infered_num_layers.pop() != num_layers:
            raise ValueError("`num_layers` must be equal to the length of `upsample_rates`, `upsample_kernel_sizes`, `nsf_kernel_sizes`, `nsf_strides`, `nsf_padding`, `nsf_residual_block_kernel_sizes`")

        if len(mrf_kernel_sizes) != len(mrf_dilation_sizes):
            raise ValueError("`mrf_kernel_sizes` and `mrf_dilation_sizes` must have the same length, correspond to the number of layers in MRF blocks.")
        
        self.hidden_size = hidden_size
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes

        self.nsf_kernel_sizes = nsf_kernel_sizes
        self.nsf_strides = nsf_strides
        self.nsf_padding = nsf_padding
        self.nsf_residual_block_kernel_sizes = nsf_residual_block_kernel_sizes

        self.mrf_kernel_sizes = mrf_kernel_sizes
        self.mrf_dilation_sizes = mrf_dilation_sizes

        super().__init__(**kwargs)


__all__ = ["HiFTNetConfig"]
