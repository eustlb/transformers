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

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import (
    AutoProcessor,
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingAudioProcessor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import require_torch, require_torchaudio


@require_torch
@require_torchaudio
class NemotronAudioProcessorMigrationTest(unittest.TestCase):
    def setUp(self):
        self.audio_processor = NemotronAsrStreamingAudioProcessor()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<blank>": 1}, unk_token="<unk>")),
            unk_token="<unk>",
            pad_token="<blank>",
        )

    def make_processor(self):
        return Nemotron3_5AsrProcessor(audio_processor=self.audio_processor, tokenizer=self.tokenizer)

    def test_legacy_constructor_and_attribute_alias(self):
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            processor = Nemotron3_5AsrProcessor(feature_extractor=self.audio_processor, tokenizer=self.tokenizer)
        self.assertIs(processor.audio_processor, self.audio_processor)
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            self.assertIs(processor.feature_extractor, self.audio_processor)
        replacement = NemotronAsrStreamingAudioProcessor()
        with self.assertWarnsRegex(FutureWarning, "audio_processor"):
            processor.feature_extractor = replacement
        self.assertIs(processor.audio_processor, replacement)
        self.assertEqual(processor.get_attributes(), ["audio_processor", "tokenizer"])
        self.assertNotIn("feature_extractor", processor.to_dict())
        with self.assertWarns(FutureWarning):
            processor = Nemotron3_5AsrProcessor(
                audio_processor=self.audio_processor, feature_extractor=replacement, tokenizer=self.tokenizer
            )
        self.assertIs(processor.audio_processor, self.audio_processor)
        self.assertIs(
            Nemotron3_5AsrProcessor(self.audio_processor, self.tokenizer).audio_processor, self.audio_processor
        )

    def test_config_migration_and_audio_output(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), TemporaryDirectory() as directory:
                processor = self.make_processor()
                processor.save_pretrained(directory)
                config_path = Path(directory) / "processor_config.json"
                config = json.loads(config_path.read_text())
                self.assertIn("audio_processor", config)
                self.assertNotIn("feature_extractor", config)
                if legacy:
                    config["feature_extractor"] = config.pop("audio_processor")
                    config_path.write_text(json.dumps(config))
                with warnings.catch_warnings(record=True) as caught:
                    restored = AutoProcessor.from_pretrained(directory)
                self.assertFalse(any("AutoFeatureExtractor" in str(w.message) for w in caught))
                self.assertEqual(restored.audio_processor.to_dict(), processor.audio_processor.to_dict())
                audio = np.zeros(16000, dtype=np.float32)
                expected = processor.audio_processor(audio, sampling_rate=16000, return_tensors="np")
                actual = restored.audio_processor(audio, sampling_rate=16000, return_tensors="np")
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
                restored.save_pretrained(directory)
                saved = json.loads(config_path.read_text())
                self.assertIn("audio_processor", saved)
                self.assertNotIn("feature_extractor", saved)

    def test_canonical_config_wins_over_legacy(self):
        with TemporaryDirectory() as directory:
            self.make_processor().save_pretrained(directory)
            config_path = Path(directory) / "processor_config.json"
            config = json.loads(config_path.read_text())
            config["feature_extractor"] = {**config["audio_processor"], "sampling_rate": 8000}
            config_path.write_text(json.dumps(config))
            restored = Nemotron3_5AsrProcessor.from_pretrained(directory)
            self.assertEqual(restored.audio_processor.sampling_rate, 16000)

    def test_audio_kwargs_use_canonical_component(self):
        from transformers.processing_utils import ProcessingKwargs

        processor = self.make_processor()
        with warnings.catch_warnings(record=True) as caught:
            kwargs = processor._merge_kwargs(ProcessingKwargs, audio_kwargs={"return_padding_mask": False})
        self.assertFalse(caught)
        self.assertIs(kwargs["audio_kwargs"]["return_padding_mask"], False)

    def test_streaming_geometry_lives_on_audio_processor(self):
        audio_processor = self.audio_processor
        self.assertEqual(audio_processor.num_lookahead_tokens, 13)
        audio_processor.set_num_lookahead_tokens(6)
        self.assertEqual(audio_processor.num_mel_frames_first_audio_chunk, 1 + 8 * 6)
        self.assertEqual(audio_processor.num_mel_frames_per_audio_chunk, 8 * 7)
        self.assertEqual(audio_processor.num_samples_first_audio_chunk, 48 * 160 + 400 // 2)
        self.assertEqual(audio_processor.num_samples_per_audio_chunk, 56 * 160 + 400)
        self.assertEqual(audio_processor.streaming_latency_ms, 560)
        self.assertEqual(audio_processor.supported_streaming_latencies_ms, {13: 1120, 6: 560, 1: 160, 0: 80})
        with self.assertRaisesRegex(ValueError, "not supported"):
            audio_processor.set_num_lookahead_tokens(5)

        audio = np.zeros(1600, dtype=np.float32)
        self.assertEqual(audio_processor(audio, sampling_rate=16000)["num_lookahead_tokens"], 6)
        self.assertEqual(
            audio_processor(audio, sampling_rate=16000, num_lookahead_tokens=1)["num_lookahead_tokens"], 1
        )
        with self.assertRaisesRegex(ValueError, "not supported"):
            audio_processor(audio, sampling_rate=16000, num_lookahead_tokens=5)
        self.assertIn("num_lookahead_tokens", audio_processor.model_input_names)

        processor = self.make_processor()
        processor.set_num_lookahead_tokens(1)
        self.assertEqual(audio_processor.num_lookahead_tokens, 1)
        self.assertEqual(processor.default_num_lookahead_tokens, 1)
        self.assertEqual(processor.supported_num_lookahead_tokens, [13, 6, 1, 0])
        self.assertEqual(processor.num_samples_per_audio_chunk, audio_processor.num_samples_per_audio_chunk)
        self.assertEqual(processor.streaming_latency_ms, audio_processor.streaming_latency_ms)

    def test_legacy_processor_level_lookahead_config(self):
        with TemporaryDirectory() as directory:
            processor = self.make_processor()
            processor.set_num_lookahead_tokens(6)
            processor.save_pretrained(directory)
            config_path = Path(directory) / "processor_config.json"
            config = json.loads(config_path.read_text())
            self.assertNotIn("default_num_lookahead_tokens", config)
            self.assertEqual(config["audio_processor"]["num_lookahead_tokens"], 6)
            self.assertEqual(AutoProcessor.from_pretrained(directory).audio_processor.num_lookahead_tokens, 6)

            # Hub checkpoints store the lookahead set at the processor level.
            del config["audio_processor"]["num_lookahead_tokens"]
            del config["audio_processor"]["supported_num_lookahead_tokens"]
            config.update(supported_num_lookahead_tokens=[13, 6, 1, 0], default_num_lookahead_tokens=1)
            config_path.write_text(json.dumps(config))
            restored = AutoProcessor.from_pretrained(directory)
            self.assertEqual(restored.audio_processor.num_lookahead_tokens, 1)

            restored.save_pretrained(directory)
            saved = json.loads(config_path.read_text())
            self.assertNotIn("default_num_lookahead_tokens", saved)
            self.assertNotIn("supported_num_lookahead_tokens", saved)
            self.assertEqual(saved["audio_processor"]["num_lookahead_tokens"], 1)
