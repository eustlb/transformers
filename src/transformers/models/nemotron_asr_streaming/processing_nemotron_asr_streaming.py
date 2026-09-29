# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


from tokenizers.decoders import DecodeStream

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class NemotronAsrStreamingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "longest",
            "return_padding_mask": True,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class NemotronAsrStreamingProcessor(ProcessorMixin):
    def __init__(
        self,
        audio_processor,
        tokenizer,
        blank_token="<blank>",
        supported_num_lookahead_tokens=None,
        default_num_lookahead_tokens=None,
    ):
        r"""
        blank_token (`str`, *optional*, defaults to `"<blank>"`):
            Blank token for RNN-T decoding.
        supported_num_lookahead_tokens (`list[int]`, *optional*):
            Legacy location of the audio processor's `supported_num_lookahead_tokens`, kept so that configs saving
            it at the processor level still load. Forwarded to the audio processor when provided.
        default_num_lookahead_tokens (`int`, *optional*):
            Legacy location of the audio processor's `num_lookahead_tokens`, kept so that configs saving it at the
            processor level still load. Forwarded to the audio processor when provided.
        """
        self.blank_token = blank_token
        self.blank_token_id = tokenizer.convert_tokens_to_ids(blank_token)
        super().__init__(audio_processor, tokenizer)
        if supported_num_lookahead_tokens is not None:
            self.audio_processor.supported_num_lookahead_tokens = supported_num_lookahead_tokens
        if default_num_lookahead_tokens is not None:
            self.audio_processor.set_num_lookahead_tokens(default_num_lookahead_tokens)

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        sampling_rate: int | None = None,
        is_streaming: bool = False,
        is_first_audio_chunk: bool | None = True,
        **kwargs: Unpack[NemotronAsrStreamingProcessorKwargs],
    ):
        r"""
        sampling_rate (`int`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate expected by the feature
            extractor (defaults to 16000 Hz). If provided, it will be validated against the processor's expected
            sampling rate, and an error will be raised if they don't match. If not provided, a warning will be
            issued and the default sampling rate will be assumed.
        is_streaming (`bool`, *optional*, defaults to `False`):
            Whether to process audio in streaming mode. When `True`, audio can be passed in chunks, using
            `is_first_audio_chunk` to distinguish the first chunk from subsequent ones.
        is_first_audio_chunk (`bool`, *optional*, defaults to `True`):
            Whether the current audio is the first chunk of a streaming session. The feature extractor uses
            `center=True` for the first chunk (and for offline use) and `center=False` for subsequent chunks,
            so that the per-chunk STFT reproduces, frame-for-frame, a single full-utterance pass. Must be
            `True` when `is_streaming=False`.

        Returns:
            [`BatchFeature`]: the audio processor (and optional tokenizer) outputs. The audio processor returns
            `num_lookahead_tokens`, the right attention context (lookahead, in subsampled encoder frames) the chunks
            were sized for (set via [`~NemotronAsrStreamingAudioProcessor.set_num_lookahead_tokens`]). Pass it to the
            model/encoder forward (or `generate`); it plays the role of Voxtral Realtime's `num_delay_tokens`.
        """
        if not is_streaming and not is_first_audio_chunk:
            raise ValueError("In non-streaming mode (`is_streaming=False`), `is_first_audio_chunk` must be `True`.")

        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            NemotronAsrStreamingProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be {output_kwargs['audio_kwargs']['sampling_rate']}, which can result in silent errors."
            )
        else:
            # Forward the caller's assertion; the audio processor resamples if it differs from its own rate.
            output_kwargs["audio_kwargs"]["sampling_rate"] = sampling_rate

        if audio is not None:
            # `center=True` for the first/offline chunk, `center=False` for subsequent streaming chunks.
            inputs = self.audio_processor(audio, center=bool(is_first_audio_chunk), **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs

        inputs["labels"] = encodings["input_ids"]
        # Prepend blank token to labels to form decoder_input_ids.
        # The RNN-T decoder expects [blank, label_0, ..., label_{U-1}] as input,
        if isinstance(text, str):
            text = [text]
        decoder_text = [self.blank_token + t for t in text]
        decoder_encodings = self.tokenizer(decoder_text, **output_kwargs["text_kwargs"])
        inputs["decoder_input_ids"] = decoder_encodings["input_ids"]
        return inputs

    @property
    def model_input_names(self):
        audio_processor_input_names = self.audio_processor.model_input_names
        return audio_processor_input_names + ["labels", "decoder_input_ids"]

    def batch_decode(self, *args, **kwargs):
        # RNN-T keeps repeated tokens (each is a separate emission), so consecutive identical tokens are not merged.
        kwargs.setdefault("group_tokens", False)
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, durations=None, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.decode`] and post-process the token-level timestamps (if
        `durations` are provided) as in the NeMo library.
        """
        # RNN-T keeps repeated tokens (each is a separate emission), so consecutive identical tokens are not merged.
        kwargs.setdefault("group_tokens", False)
        decoded = self.tokenizer.decode(*args, **kwargs)

        if durations is not None:
            token_ids = args[0]
            # Derive per-step frame indices from cumulative sum of durations.
            timestamps = durations.cumsum(dim=-1) - durations

            frame_rate = (
                self.audio_processor.spectrogram_config.stft_config.hop_length
                / self.audio_processor.sampling_rate
                * self.audio_processor.subsampling_factor
            )
            # Filter padding/blank tokens and decode per sequence to keep track of token-level timestamps
            # See `compute_rnnt_timestamps` in NeMo:
            # https://github.com/NVIDIA-NeMo/NeMo/blob/1692a8fb97e1aadc883cfadd2a57c4e8a1b793aa/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L993
            skip_ids = {self.tokenizer.pad_token_id, self.blank_token_id}
            proc_timestamps = []
            for batch_ids, batch_timestamps in zip(token_ids, timestamps):
                stream = DecodeStream(skip_special_tokens=True)
                timestamp_dict = []
                for i, token_id in enumerate(batch_ids):
                    if int(token_id) in skip_ids:
                        continue
                    chunk = stream.step(self.tokenizer._tokenizer, int(token_id))
                    if chunk is not None:
                        # RNN-T tokens each span a single frame (their per-step value is a 0/1 encoder advance,
                        # not a span).
                        start = int(batch_timestamps[i])
                        timestamp_dict.append(
                            {
                                "token": chunk,
                                "start": start,
                                "end": start + 1,
                            }
                        )
                proc_timestamps.append(self._refine_timestamps(timestamp_dict, frame_rate))

            return decoded, proc_timestamps
        return decoded

    def _refine_timestamps(self, char_offsets, frame_rate):
        # RNN-T mirrors NeMo's raw char-level timestamps, which keep every token (punctuation included) at its
        # own emitted frame. Only convert frame indices to seconds.
        for offset in char_offsets:
            offset["start"] = offset["start"] * frame_rate
            offset["end"] = offset["end"] * frame_rate
        return char_offsets

    # Streaming geometry lives on the audio processor; these forward to it.
    @property
    def supported_num_lookahead_tokens(self) -> list[int]:
        return self.audio_processor.supported_num_lookahead_tokens

    @property
    def default_num_lookahead_tokens(self) -> int:
        return self.audio_processor.num_lookahead_tokens

    def set_num_lookahead_tokens(self, num_lookahead_tokens: int):
        """Forwards to the audio processor's `set_num_lookahead_tokens`."""
        self.audio_processor.set_num_lookahead_tokens(num_lookahead_tokens)

    @property
    def streaming_latency_ms(self) -> int:
        return self.audio_processor.streaming_latency_ms

    @property
    def supported_streaming_latencies_ms(self) -> dict[int, int]:
        return self.audio_processor.supported_streaming_latencies_ms

    @property
    def num_mel_frames_first_audio_chunk(self) -> int:
        return self.audio_processor.num_mel_frames_first_audio_chunk

    @property
    def num_mel_frames_per_audio_chunk(self) -> int:
        return self.audio_processor.num_mel_frames_per_audio_chunk

    @property
    def num_samples_first_audio_chunk(self) -> int:
        return self.audio_processor.num_samples_first_audio_chunk

    @property
    def num_samples_per_audio_chunk(self) -> int:
        return self.audio_processor.num_samples_per_audio_chunk


__all__ = ["NemotronAsrStreamingProcessor"]
