# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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
"""HiggsAudioTokenizerModel model."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ...utils import auto_docstring, can_return_tuple
from ..auto.modeling_auto import AutoModel
from ..dac.modeling_dac import DacDecoder, DacDecoderBlock, DacEncoder
from ..xcodec.modeling_xcodec import (
    SemanticEncoder,
    XcodecDecoderOutput,
    XcodecEncoderOutput,
    XcodecEuclideanCodebook,
    XcodecOutput,
    XcodecPreTrainedModel,
    XcodecResidualVectorQuantization,
)
from .configuration_higgs_audio_tokenizer import HiggsAudioTokenizerConfig


class HiggsAudioTokenizerAcousticEncoder(DacEncoder):
    def __init__(self, config):
        super().__init__(config)
        d_model = config.encoder_hidden_size * 2**len(config.downsampling_ratios)
        self.conv2 = nn.Conv1d(d_model, config.acoustic_hidden_size, kernel_size=3, padding=1)


class HiggsAudioTokenizerSemanticEncoder(SemanticEncoder): ...


class HiggsAudioTokenizerEncoder(nn.Module):
    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__()
        self.acoustic_encoder = HiggsAudioTokenizerAcousticEncoder(config)
        self.semantic_encoder = HiggsAudioTokenizerSemanticEncoder(config)
        self.semantic_model = AutoModel.from_config(config.semantic_config)

        self.sample_rate = config.sample_rate
        self.semantic_sample_rate = config.semantic_sample_rate

    def _extract_semantic_features(self, input_values):
        with torch.no_grad():
            input_values = torchaudio.functional.resample(
                input_values, self.config.sampling_rate, self.config.semantic_sample_rate
            )
            input_values = input_values[:, 0, :]
            input_values = F.pad(input_values, (self.config.pad, self.config.pad))
            outputs = self.semantic_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            stacked = torch.stack(hidden_states, dim=1)
            semantic_features = stacked.mean(dim=1)
            semantic_features = semantic_features[:, :: self.config.semantic_downsample_factor, :]
            return semantic_features

    def forward(self, input_values: torch.Tensor):
        acoustic_embeds = self.acoustic_encoder(input_values)

        semantic_features = self._extract_semantic_features(input_values)
        semantic_embeds = self.semantic_encoder(semantic_features.transpose(1, 2))

        input_embeds = torch.cat([acoustic_embeds, semantic_embeds], dim=1)
        return input_embeds


class HiggsAudioTokenizerDecoderBlock(DacDecoderBlock):
    def __init__(self, config, stride: int = 1, stride_index: int = 1):
        super().__init__(config)
        input_dim = config.decoder_hidden_size // 2**stride_index
        output_dim = config.decoder_hidden_size // 2 ** (stride_index + 1)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=(stride % 2,),
        )

class HiggsAudioTokenizerDecoder(DacDecoder):
    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__(config)
        input_channel = config.acoustic_hidden_size
        del self.tanh

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        for layer in self.block:
            hidden_states = layer(hidden_states)

        hidden_states = self.snake1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class HiggsAudioTokenizerEuclideanCodebook(XcodecEuclideanCodebook): ...


class HiggsAudioTokenizerVectorQuantization(nn.Module):
    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__()
        self.codebook = HiggsAudioTokenizerEuclideanCodebook(config)
        self.project_in = nn.Linear(config.hidden_size, config.codebook_dim)
        self.project_out = nn.Linear(config.codebook_dim, config.hidden_size)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.project_in(hidden_states)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class HiggsAudioTokenizerResidualVectorQuantization(XcodecResidualVectorQuantization): ...


class HiggsAudioTokenizerPreTrainedModel(XcodecPreTrainedModel): ...


class HiggsAudioTokenizerEncoderOutput(XcodecEncoderOutput): ...


class HiggsAudioTokenizerDecoderOutput(XcodecDecoderOutput): ...


class HiggsAudioTokenizerOutput(XcodecOutput): ...


@auto_docstring(custom_intro="""The Higgs Audio neural audio codec model.""")
class HiggsAudioTokenizerModel(HiggsAudioTokenizerPreTrainedModel):
    def __init__(self, config: HiggsAudioTokenizerConfig):
        super().__init__(config)
        self.encoder = HiggsAudioTokenizerEncoder(config)
        self.decoder = HiggsAudioTokenizerDecoder(config)
        self.quantizer = HiggsAudioTokenizerResidualVectorQuantization(config)
        self.encoder_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder_proj = nn.Linear(config.hidden_size, config.acoustic_hidden_size)

    @can_return_tuple
    @auto_docstring
    def encode(
        self,
        input_values: torch.Tensor,
        bandwidth: Optional[float] = None,
        **kwargs,
    ) -> Union[torch.Tensor, HiggsAudioTokenizerEncoderOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            Float values of the input audio waveform.
        bandwidth (`float`, *optional*):
            The target bandwidth in (kbps) supports only values in `config.target_bandwidths`.
            Defaults to the highest available bandwidth `4.0` kbps.

        Returns:
            `torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)` containing the discrete encoded audio codes.
        """
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[-1]
        elif bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. Select one of {self.config.target_bandwidths}."
            )

        inputs_embeds = self.encoder(input_values)
        inputs_embeds = self.encoder_proj(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        audio_codes = self.quantizer.encode(inputs_embeds, bandwidth)
        audio_codes = audio_codes.transpose(0, 1)

        return HiggsAudioTokenizerEncoderOutput(audio_codes)

    @can_return_tuple
    @auto_docstring
    def decode(
        self, audio_codes: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, HiggsAudioTokenizerDecoderOutput]:
        r"""
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`):
            Discrete code indices computed using `model.encode`.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`]

        Returns:
            Decoded audio values of shape `(batch_size, channels, num_samples)` obtained using the decoder part of HiggsAudioTokenizerModel.
        """
        audio_codes = audio_codes.transpose(0, 1)
        quantized = self.quantizer.decode(audio_codes)
        quantized_acoustic = self.decoder_proj(quantized.transpose(1, 2)).transpose(1, 2)
        audio_values = self.decoder(quantized_acoustic)

        return HiggsAudioTokenizerDecoderOutput(audio_values)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        audio_codes: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], HiggsAudioTokenizerOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, num_samples)`):
            The raw float values of the input audio waveform.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discrete code indices computed using `model.encode`.
        bandwidth (`float`, *optional*):
            Target bandwidth in kbps. Must be one of `config.target_bandwidths`.
            Defaults to the highest available bandwidth.
        return_dict (`bool`, *optional*):
            Whether to return a [`HiggsAudioTokenizerOutput`] instead of a plain tuple.

        Returns:
            `HiggsAudioTokenizerOutput` or tuple `(audio_codes, audio_values)`:
            - `audio_codes` of shape `(batch_size, num_quantizers, codes_length)`: the quantized discrete codes.
            - `audio_values` of shape `(batch_size, channels, num_samples)`: the reconstructed audio waveform given the codes.

        Example:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, HiggsAudioTokenizerModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "bosonai/higgs-audio-v2-tokenizer"
        >>> model = HiggsAudioTokenizerModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```
        """
        length = input_values.shape[-1]

        if audio_codes is None:
            audio_codes = self.encode(input_values, bandwidth, return_dict=False)

        audio_values = self.decode(audio_codes, return_dict=return_dict)[0][..., :length]

        return HiggsAudioTokenizerOutput(audio_codes=audio_codes, audio_values=audio_values)


__all__ = ["HiggsAudioTokenizerModel", "HiggsAudioTokenizerPreTrainedModel"]
