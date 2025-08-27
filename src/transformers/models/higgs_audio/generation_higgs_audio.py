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
"""HiggsAudioGenerationMixin."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput
from ...utils import ModelOutput


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""

    TEXT = 0  # Text generation mode
    AUDIO_INIT = 1  # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2  # Audio generation mode in progress


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=5. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  P,  P,  P]
    - [ B,  *,  *,  *,  *,  *,  P,  P]
    - [ B,  B,  *,  *,  *,  *,  *,  P]
    - [ B,  B,  B,  *,  *,  *,  *,  *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids, input_ids_with_gen_mask


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


def merge_input_ids_with_audio_features(
    audio_in_embed,
    audio_in_ids_start,
    audio_out_embed,
    audio_out_ids_start,
    audio_in_token_idx,
    audio_out_token_idx,
    inputs_embeds,
    input_ids,
    attention_mask,
    label_ids,
    pad_token_id,
    ignore_index=-100,
    round_to=8,
    left_padding=True,
):
    """
    Merge input_ids with audio features into final embeddings.

    Args:
        audio_in_embed (`torch.Tensor` of shape `(total_num_audio_in_tokens, embed_dim)`):
            The embeddings of audio-in tokens
        audio_in_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-in tokens for each audio
        audio_out_embed (`torch.Tensor` of shape `(total_num_audio_out_tokens, embed_dim)`):
            The embeddings of audio-out tokens
        audio_out_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-out tokens for each audio
        audio_in_token_idx
            The index of the audio-in token in the vocabulary
        audio_out_token_idx
            The index of the audio-out token in the vocabulary
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with audio embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with audio token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        label_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            labels need to be recalculated to support training (if provided)
        pad_token_id (`int`):
            The index of the pad token in the vocabulary
        ignore_index
            The index to ignore in the loss calculation
        round_to
            The number to round to for padding
        left_padding
            Whether to apply left padding

    Returns:
        final_embedding
            The final embeddings after merging audio embeddings with text embeddings.
        final_attention_mask
            The final attention mask after merging audio embeddings with text embeddings.
        final_labels
            The labels for the text stream
        position_ids
            Positional ids for the merged data
        final_input_ids
            The final input_ids after merging audio embeddings with text embeddings.
        final_audio_in_mask
            Mask for audio-in embeddings
        final_audio_in_discrete_codes_mask
            Mask for audio-in discrete tokens
        final_audio_out_mask
            Mask for audio-out embeddings

    Explanation:
        each audio has variable length embeddings, with length specified by
        - audio_in_ids_start
        - audio_out_ids_start

        Task:
        - fill each <|AUDIO|> with audio embeddings from audio codebooks
        - fill each <|AUDIO_OUT|> with the audio-out embeddings

        Example:
            <|AUDIO_OUT|>: X (5 tokens), Y (3 tokens)
            <|AUDIO|>: Z (8 tokens)

            X, Y are in the same sequence (in-context voice-clone). Z is in a different sequence (audio understanding).
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]

    """
    if label_ids is None:
        skip_labels = True
    else:
        skip_labels = False
    if audio_in_embed is not None and audio_in_embed.shape[0] == 0:
        audio_in_embed = None
    if audio_out_embed is not None and audio_out_embed.shape[0] == 0:
        audio_out_embed = None

    batch_size, sequence_length, embed_dim = inputs_embeds.shape

    target_device = inputs_embeds.device
    if left_padding is None:
        left_padding = torch.any(attention_mask[:, 0] == 0)

    audio_in_token_mask = input_ids == audio_in_token_idx
    audio_out_token_mask = input_ids == audio_out_token_idx
    text_token_mask = (input_ids != audio_in_token_idx) & (input_ids != audio_out_token_idx)

    # 1. Calculate the number of tokens for each placeholder (like [<|AUDIO|>, <|AUDIO_OUT|>]).
    token_placeholder_num = torch.ones_like(input_ids)

    if audio_in_embed is not None:
        audio_in_codes_length = torch.concat(
            [
                audio_in_ids_start[1:] - audio_in_ids_start[:-1],
                torch.tensor(
                    [audio_in_embed.shape[0] - audio_in_ids_start[-1]],
                    device=audio_in_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_in_token_mask] = audio_in_codes_length.long()

    if audio_out_embed is not None:
        audio_out_codes_length = torch.concat(
            [
                audio_out_ids_start[1:] - audio_out_ids_start[:-1],
                torch.tensor(
                    [audio_out_embed.shape[0] - audio_out_ids_start[-1]],
                    device=audio_out_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_out_token_mask] = audio_out_codes_length.long()

    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = _ceil_to_nearest(token_placeholder_num.sum(-1).max(), round_to)
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions += nb_audio_pad[:, None]  # offset for left padding

    # 2. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        (batch_size, max_token_num, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        (batch_size, max_token_num), dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    final_input_ids = torch.full(
        (batch_size, max_token_num), pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
    )
    if skip_labels:
        final_labels = None
    else:
        final_labels = torch.full(
            (batch_size, max_token_num), ignore_index, dtype=label_ids.dtype, device=inputs_embeds.device
        )

    final_audio_in_mask = torch.full((batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device)
    final_audio_in_discrete_codes_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    final_audio_out_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    # 3. Get the audio-in token positions and audio-out token positions
    batch_id = torch.arange(batch_size, device=target_device).unsqueeze(1).expand(batch_size, sequence_length)
    audio_in_batch_id = batch_id[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_batch_id = batch_id[audio_out_token_mask]  # Shape (num_audio_out,)
    audio_features_token_ends = new_token_positions[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_embed_ends = new_token_positions[audio_out_token_mask]  # Shape (num_audio_out,)

    if audio_in_embed is not None:
        # Fill in the audio-in embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_in_ids_start.shape[0], max_token_num)
        )
        audio_in_embed_token_starts = audio_features_token_ends - audio_in_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_in_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_in_embed
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True
        final_audio_in_discrete_codes_mask[batch_indices, col_indices] = True

    if audio_out_embed is not None:
        # Fill in the audio-out embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_out_ids_start.shape[0], max_token_num)
        )
        audio_out_embed_token_starts = audio_out_embed_ends - audio_out_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_out_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_out_embed_ends.unsqueeze(1))
        )
        batch_indices = audio_out_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_out_embed
        final_input_ids[batch_indices, col_indices] = audio_out_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_out_mask[batch_indices, col_indices] = True

    # Fill in the original text embeddings and labels
    batch_indices, non_audio_indices = torch.where(text_token_mask)
    text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
    if not skip_labels:
        final_labels[batch_indices, text_to_overwrite] = label_ids[batch_indices, non_audio_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
    final_attention_mask = final_attention_mask | final_audio_in_mask | final_audio_out_mask

    # Trim the tensor if there are redundant padding tokens
    if left_padding:
        first_non_zero_loc = final_attention_mask.sum(0).nonzero()[0]
        first_non_zero_loc = (first_non_zero_loc // round_to) * round_to
        if first_non_zero_loc > 0:
            final_attention_mask = final_attention_mask[:, first_non_zero_loc:]
            final_embedding = final_embedding[:, first_non_zero_loc:]
            if not skip_labels:
                final_labels = final_labels[:, first_non_zero_loc:]
            final_input_ids = final_input_ids[:, first_non_zero_loc:]
            final_audio_in_mask = final_audio_in_mask[:, first_non_zero_loc:]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, first_non_zero_loc:]
            final_audio_out_mask = final_audio_out_mask[:, first_non_zero_loc:]
    else:
        # We have done right padding, so we need to trim the mask
        last_non_zero_loc = final_attention_mask.sum(0).nonzero()[-1] + 1
        last_non_zero_loc = ((last_non_zero_loc + round_to - 1) // round_to) * round_to
        if last_non_zero_loc < max_token_num:
            final_attention_mask = final_attention_mask[:, :last_non_zero_loc]
            final_embedding = final_embedding[:, :last_non_zero_loc]
            if not skip_labels:
                final_labels = final_labels[:, :last_non_zero_loc]
            final_input_ids = final_input_ids[:, :last_non_zero_loc]
            final_audio_in_mask = final_audio_in_mask[:, :last_non_zero_loc]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, :last_non_zero_loc]
            final_audio_out_mask = final_audio_out_mask[:, :last_non_zero_loc]

    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        position_ids,
        final_input_ids,
        final_audio_in_mask,
        final_audio_in_discrete_codes_mask,
        final_audio_out_mask,
    )


@dataclass
class HiggsAudioGenerationOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.model.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.model.audio_codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
    """

    audio_sequences: Optional[list[torch.LongTensor]] = None


class HiggsAudioGenerationMixin(GenerationMixin):
    def _sample_audio_tokens(
        self,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""

        # parameters related to repetition aware sampling
        ras_win_len = generation_config.ras_win_len
        ras_win_max_num_repeat = generation_config.ras_win_max_num_repeat
        audio_eos_token_id = generation_config.audio_eos_token_id
        # In the audio generation mode, we sample from audio_logits and keep updating audio_out_ids.
        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        # TopP, TopK logits processor supports empty input_ids
        next_audio_token_scores = logits_processor(None, next_audio_token_logits)

        # token selection
        if do_sample:
            # next_audio_token_scores has been applied top_p, top_k, and temperature.
            probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
            next_audio_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        # next_tokens: (num_codebooks, )
        if ras_win_len is not None:
            # check if there are repetitions over a window of tokens.
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

            # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
            row_indices = torch.nonzero(rep_num >= ras_win_max_num_repeat).squeeze(1)
            resampled_next_tokens = (
                next_audio_token_logits[row_indices]
                .softmax(dim=-1)
                .multinomial(1, replacement=True, generator=torch_generator)
                .squeeze(1)
            )
            next_audio_tokens[row_indices] = resampled_next_tokens

        # Force the next text tokens to be <|AUDIO_OUT|> in audio generation mode
        next_tokens = torch.full(
            (audio_logits.shape[0],),
            self.config.audio_out_token_idx,
            dtype=torch.long,
            device=device,
        )

        # Handle delay_pattern
        if self.model.use_delay_pattern:
            if num_delay + 1 < next_audio_tokens.shape[0]:
                next_audio_tokens[(num_delay + 1) :] = self.config.audio_stream_bos_id
                num_delay += 1
            if num_remaining_delays is not None:
                next_audio_tokens[: (self.config.audio_num_codebooks - num_remaining_delays)] = (
                    self.config.audio_stream_eos_id
                )
                num_remaining_delays -= 1
            else:
                all_eos_indices = (next_audio_tokens == self.config.audio_stream_eos_id).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    next_audio_tokens[:last_eos_idx] = self.config.audio_stream_eos_id
                    num_remaining_delays = self.config.audio_num_codebooks - last_eos_idx - 1
            if num_remaining_delays is not None and num_remaining_delays <= 0:
                next_tokens[...] = audio_eos_token_id
                num_delay = 0
                num_remaining_delays = None

        return (
            next_tokens,
            next_audio_tokens,
            next_audio_token_logits,
            next_audio_token_scores,
            num_delay,
            num_remaining_delays,
        )

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        generation_mode: GenerationMode,
        torch_generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.config.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config.audio_num_codebooks,),
                self.config.audio_stream_bos_id,
                dtype=torch.long,
                device=device,
            )
        else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_audio_tokens = None

        return next_tokens, next_audio_tokens, next_token_logits, next_token_scores

    # Overwrite GenerationMixin._update_model_kwargs_for_generation
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True,
    ) -> dict[str, Any]:
        """Update the model kwargs for each step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if extend_attention_mask:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        if "cache_audio_discrete_codes_mask" in model_kwargs:
            if model_kwargs["cache_audio_discrete_codes_mask"] is None:
                model_kwargs["cache_audio_discrete_codes_mask"] = (
                    outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask
                )
            else:
                model_kwargs["cache_audio_discrete_codes_mask"] = torch.concat(
                    [
                        model_kwargs["cache_audio_discrete_codes_mask"],
                        outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask,
                    ],
                    1,
                )

        return model_kwargs

    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for joint text/audio models using **multinomial sampling**.

        This function may also be revised to support generating samples from HiggsAudio-like end-to-end text/audio models built on top of LLMs.
        If the input_ids ends with <|audio_out_bos|>, we will switch to the audio-generation mode.

        ```
        ...<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>
        ```

        Otherwise, we will keep generating the text tokens.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        assert input_ids.shape[0] == 1, "Only support batch_size=1 in _sample()"
        audio_out_bos_token_id = generation_config.audio_out_bos_token_id

        # torch generator for sampling
        seed = generation_config.seed
        if seed is not None:
            torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        else:
            torch_generator = None

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        # Used to track which past_key_value
        self.current_past_key_values_bucket = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if generation_config.use_cache:
            model_kwargs["cache_audio_discrete_codes_mask"] = None

        init_model_input = True
        num_delay = 0
        num_remaining_delays = None
        audio_sequences = []
        # A tensor to keep track of all the audio placeholder tokens.
        input_ids_full = input_ids.clone()

        # Initialize the audio variables based on the input prompt.
        if input_ids[0][-1] == self.config.audio_out_token_idx:
            audio_sequences = [model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1] :]]
            if self.model.use_delay_pattern:
                num_delay = (
                    self.config.audio_num_codebooks
                    - (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_bos_id).sum()
                )
                all_eos_indices = (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_eos_id).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    num_remaining_delays = self.config.audio_num_codebooks - last_eos_idx - 1

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # Check which multimodal stage we are in
            # FIXME: Assume single input generation
            if input_ids[0][-1] == audio_out_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.config.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
            else:
                generation_mode = GenerationMode.TEXT

            is_audio_generation_mode = generation_mode == GenerationMode.AUDIO_IN_PROGRESS

            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if is_audio_generation_mode and generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor([0], dtype=torch.long, device=input_ids.device)
                elif not is_audio_generation_mode:
                    del model_inputs["audio_out_ids"]
                    del model_inputs["audio_out_ids_start"]

                if generation_config.use_cache:
                    if "audio_in_ids" in model_inputs and model_inputs["audio_in_ids"] is not None:
                        model_inputs["audio_in_ids"] = None
                        model_inputs["audio_in_ids_start"] = None

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                extend_attention_mask=True,
            )

            # After the first forward pass, we can set init_model_input to False.
            init_model_input = False

            if synced_gpus and this_peer_finished:
                continue

            if is_audio_generation_mode:
                # In audio generation mode, we sample the audio tokens from audio logits.
                # It might also generate the audio eos token to end the audio generation.
                (
                    next_tokens,
                    next_audio_tokens,
                    next_audio_token_logits,
                    next_audio_token_scores,
                    num_delay,
                    num_remaining_delays,
                ) = self._sample_audio_tokens(
                    audio_logits=outputs.audio_logits,
                    audio_out_ids=model_kwargs["audio_out_ids"],
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    torch_generator=torch_generator,
                    generation_config=generation_config,
                    num_delay=num_delay,
                    num_remaining_delays=num_remaining_delays,
                )

                # update generated ids, model inputs, and length for next step
                model_kwargs["audio_out_ids"] = torch.cat(
                    [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
                )
                audio_sequences[-1] = torch.cat([audio_sequences[-1], next_audio_tokens[:, None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_audio_tokens.cpu())
            else:
                # In text generation mode, we sample the text tokens from text logits.
                # It might also generate the audio placeholder token to start the audio generation.
                next_tokens, next_audio_tokens, next_token_logits, next_token_scores = self._sample_text_tokens(
                    input_ids=input_ids,
                    logits=outputs.logits,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    generation_mode=generation_mode,
                    torch_generator=torch_generator,
                )

                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                if next_audio_tokens is not None:
                    # If the token is audio bos token, we will generate the audio placeholder token
                    # and the corrensponding audio stream bos token to start the audio generation.
                    audio_sequences.append(next_audio_tokens[:, None])
                    if streamer is not None:
                        streamer.put(next_audio_tokens.cpu())
                    if model_kwargs["audio_out_ids"] is None or model_kwargs["audio_out_ids"].shape[0] == 0:
                        # Initialize audio_out_ids
                        model_kwargs["audio_out_ids"] = next_audio_tokens[:, None]
                        model_kwargs["audio_out_ids_start"] = torch.tensor(
                            [0], dtype=torch.long, device=input_ids.device
                        )
                    else:
                        model_kwargs["audio_out_ids_start"] = torch.concat(
                            [
                                model_kwargs["audio_out_ids_start"],
                                torch.tensor(
                                    [model_kwargs["audio_out_ids"].shape[1]], dtype=torch.long, device=input_ids.device
                                ),
                            ],
                            dim=0,
                        )
                        model_kwargs["audio_out_ids"] = torch.concat(
                            [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=1
                        )

            if return_dict_in_generate:
                if output_scores:
                    if is_audio_generation_mode:
                        scores += (next_audio_token_scores,)
                    else:
                        scores += (next_token_scores,)
                if output_logits:
                    if is_audio_generation_mode:
                        raw_logits += (next_audio_token_logits,)
                    else:
                        raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if hasattr(generation_config, "tokenizer_length"):
                tokenizer_length = generation_config.tokenizer_length
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(
                        f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior."
                    )

            # update generated ids, model inputs, and length for next step
            if not is_audio_generation_mode or next_tokens[0] != self.config.audio_out_token_idx:
                # We only add one <|AUDIO_OUT|> token to the input_ids for simplicity.
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                audio_sequences=audio_sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        audio_out_bos_token_id: Optional[int] = None,
        audio_eos_token_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        The generate function in huggingface generally follows these steps:

        for sample_step in 1, 2, 3, 4, 5, ...
            ...

        """
        assert input_ids.shape[0] == 1, (
            "Currently HiggsAudioModel.generate() only supports batch_size=1. See the implementation of "
        )
        generation_config, kwargs = self._prepare_generation_config(kwargs.pop("generation_config", None), **kwargs)
        if audio_out_bos_token_id is not None:
            generation_config.audio_out_bos_token_id = audio_out_bos_token_id
        else:
            try:
                generation_config.audio_out_bos_token_id = self.config.audio_out_bos_token_id
            except AttributeError:
                generation_config.audio_out_bos_token_id = None

        if audio_eos_token_id is not None:
            generation_config.audio_eos_token_id = audio_eos_token_id
        else:
            try:
                generation_config.audio_eos_token_id = self.config.audio_eos_token_id
            except AttributeError:
                generation_config.audio_eos_token_id = None

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

        generation_config.ras_win_len = kwargs.pop("ras_win_len", 7)
        generation_config.ras_win_max_num_repeat = kwargs.pop("ras_win_max_num_repeat", 2)
        # Set generation seed if determinstic generation is required
        if seed is not None:
            generation_config.seed = seed
        else:
            generation_config.seed = None

        # Store tokenizer in generation config if it is in kwargs without popping it
        if "tokenizer" in kwargs:
            generation_config.tokenizer_length = len(kwargs["tokenizer"])

        # input_ids: [bsz, seq_len]
        # The merging of audio features happens inside the forward path. The input_ids does not need to change.
        input_ids_length = input_ids.shape[-1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=None,
            inputs_tensor=None,
            input_ids_length=input_ids_length,
        )
        assert generation_config.num_beams == 1, "Currently, we only support beam search with num_beams=1"
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_scores = generation_config.output_scores

        # When attn_implement is spda or flash-attention, it will create causal mask automatically.
        attention_mask = kwargs.pop("attention_mask", None)
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs,
        )
