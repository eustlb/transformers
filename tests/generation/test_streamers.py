# Copyright 2023 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from queue import Empty
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest

from transformers import (
    AsyncTextIteratorStreamer,
    AudioIteratorStreamer,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
    is_torch_available,
)
from transformers.testing_utils import CaptureStdout, require_torch, require_torchaudio, torch_device
from transformers.utils.logging import _get_library_root_logger

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, NemotronAsrStreamingAudioProcessor


@require_torch
class StreamerTester(unittest.TestCase):
    def test_text_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, greedy_text)

    def test_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    def test_text_streamer_skip_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        new_greedy_ids = greedy_ids[:, input_ids.shape[1] :]
        new_greedy_text = tokenizer.decode(new_greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, new_greedy_text)

    def test_text_streamer_decode_kwargs(self):
        # Tests that we can pass `decode_kwargs` to the streamer to control how the tokens are decoded. Must be tested
        # with actual models -- the dummy models' tokenizers are not aligned with their models, and
        # `skip_special_tokens=True` has no effect on them
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = torch.ones((1, 5), device=torch_device).long() * model.config.bos_token_id

        root = _get_library_root_logger()
        with patch.object(root, "propagate", False):
            with CaptureStdout() as cs:
                streamer = TextStreamer(tokenizer, skip_special_tokens=True)
                model.generate(input_ids, max_new_tokens=1, do_sample=False, streamer=streamer)

        # The prompt contains a special token, so the streamer should not print it. As such, the output text, when
        # re-tokenized, must only contain one token
        streamer_text = cs.out[:-1]  # Remove the final "\n"
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors="pt")
        self.assertEqual(streamer_text_tokenized.input_ids.shape, (1, 1))

    def test_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so an exception will be raised
        with self.assertRaises(Empty):
            streamer_text = ""
            for new_text in streamer:
                streamer_text += new_text


@require_torch
@pytest.mark.asyncio(loop_scope="class")
class AsyncStreamerTester(unittest.IsolatedAsyncioTestCase):
    async def test_async_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        streamer = AsyncTextIteratorStreamer(tokenizer)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        async for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    async def test_async_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        streamer = AsyncTextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so TimeoutError will be raised
        with self.assertRaises(TimeoutError):
            streamer_text = ""
            async for new_text in streamer:
                streamer_text += new_text


@require_torch
@require_torchaudio
class AudioIteratorStreamerTester(unittest.TestCase):
    def test_audio_processor_chunks_match_manual_loop(self):
        audio_processor = NemotronAsrStreamingAudioProcessor()
        audio_processor.set_num_lookahead_tokens(6)
        audio = np.random.RandomState(0).randn(3 * 16000).astype(np.float32)

        streamer = AudioIteratorStreamer(audio_processor)
        self.assertEqual(streamer.model_kwargs, {"num_lookahead_tokens": 6})

        def produce():
            for start in range(0, len(audio), 1234):
                streamer.put(audio[start : start + 1234])
            streamer.end()

        thread = Thread(target=produce)
        thread.start()
        chunks = list(streamer)
        thread.join()

        # The manual chunking loop from the model docs.
        stft_config = audio_processor.spectrogram_config.stft_config
        num_frames = audio_processor.num_mel_frames_first_audio_chunk
        first_chunk = audio[: audio_processor.num_samples_first_audio_chunk]
        expected = [audio_processor(first_chunk, center=True, return_tensors="pt").input_features[:, :num_frames]]
        start = num_frames * stft_config.hop_length - stft_config.n_fft // 2
        while (end := start + audio_processor.num_samples_per_audio_chunk) < len(audio):
            expected.append(audio_processor(audio[start:end], center=False, return_tensors="pt").input_features)
            num_frames += audio_processor.num_mel_frames_per_audio_chunk
            start = num_frames * stft_config.hop_length - stft_config.n_fft // 2

        self.assertEqual(len(chunks), len(expected))
        for chunk, expected_chunk in zip(chunks, expected):
            torch.testing.assert_close(chunk, expected_chunk, rtol=0, atol=0)
