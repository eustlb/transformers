# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Owlv2 model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
import requests

from transformers import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Owlv2ForObjectDetection, Owlv2Model, Owlv2TextModel, Owlv2VisionModel


if is_vision_available():
    from PIL import Image

    from transformers import OwlViTProcessor


# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTVisionModelTester with OwlViT->Owlv2
class Owlv2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=32,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Owlv2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = Owlv2VisionModel(config=config).to(torch_device)
        model.eval()

        pixel_values = pixel_values.to(torch.float32)

        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTVisionModelTest with OwlViT->Owlv2, OWL-ViT->OwlV2, OWLVIT->OWLV2, owlvit-base-patch32->owlv2-base-patch16-ensemble
class Owlv2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as OWLV2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Owlv2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Owlv2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Owlv2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="OWLV2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlv2-base-patch16-ensemble"
        model = Owlv2VisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTTextModelTester with OwlViT->Owlv2
class Owlv2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        num_queries=4,
        seq_length=16,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=16,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size * self.num_queries, self.seq_length], self.vocab_size)
        input_mask = None

        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size * self.num_queries, self.seq_length])

        if input_mask is not None:
            num_text, seq_length = input_mask.shape

            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(num_text,))
            for idx, start_index in enumerate(rnd_start_indices):
                input_mask[idx, :start_index] = 1
                input_mask[idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Owlv2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Owlv2TextModel(config=config).to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=input_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size * self.num_queries, self.seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size * self.num_queries, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTTextModelTest with OwlViT->Owlv2, OWL-ViT->OwlV2, OWLVIT->OWLV2, owlvit-base-patch32->owlv2-base-patch16-ensemble
class Owlv2TextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2TextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Owlv2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Owlv2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="OWLV2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlv2-base-patch16-ensemble"
        model = Owlv2TextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class Owlv2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Owlv2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Owlv2VisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Owlv2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = Owlv2Model(config).to(torch_device).eval()

        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        image_logits_size = (
            self.vision_model_tester.batch_size,
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
        )
        text_logits_size = (
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
            self.vision_model_tester.batch_size,
        )
        self.parent.assertEqual(result.logits_per_image.shape, image_logits_size)
        self.parent.assertEqual(result.logits_per_text.shape, text_logits_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": False,
        }
        return config, inputs_dict


@require_torch
# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTModelTest with OwlViT->Owlv2, OWL-ViT->OwlV2, OWLVIT->OWLV2, owlvit-base-patch32->owlv2-base-patch16-ensemble
class Owlv2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2Model,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Owlv2Model,
            "zero-shot-object-detection": Owlv2ForObjectDetection,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = Owlv2ModelTester(self)
        common_properties = ["projection_dim", "logit_scale_init_value"]
        self.config_tester = ConfigTester(
            self, config_class=Owlv2Config, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Owlv2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initialization is different for OWLV2
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initialized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init).to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # OWLV2 needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            loaded_model = loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict:
                if key not in model_state_dict:
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Owlv2Config and check if we can load Owlv2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Owlv2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Owlv2Config and check if we can load Owlv2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Owlv2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlv2-base-patch16-ensemble"
        model = Owlv2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTForObjectDetectionTester with OwlViT->Owlv2, OWL-ViT->OwlV2, OWLVIT->OWLV2
class Owlv2ForObjectDetectionTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = Owlv2TextModelTester(parent)
        self.vision_model_tester = Owlv2VisionModelTester(parent)
        self.is_training = is_training
        self.text_config = self.text_model_tester.get_config().to_dict()
        self.vision_config = self.vision_model_tester.get_config().to_dict()
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, pixel_values, input_ids, attention_mask

    def get_config(self):
        return Owlv2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            projection_dim=64,
        )

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        model = Owlv2ForObjectDetection(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        pred_boxes_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_logits_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_class_embeds_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            self.text_model_tester.hidden_size,
        )
        self.parent.assertEqual(result.pred_boxes.shape, pred_boxes_size)
        self.parent.assertEqual(result.logits.shape, pred_logits_size)
        self.parent.assertEqual(result.class_embeds.shape, pred_class_embeds_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
# Copied from tests.models.owlvit.test_modeling_owlvit.OwlViTForObjectDetectionTest with OwlViT->Owlv2, OWL-ViT->OwlV2, OWLVIT->OWLV2, owlvit-base-patch32->owlv2-base-patch16-ensemble
class Owlv2ForObjectDetectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Owlv2ForObjectDetection,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = Owlv2ForObjectDetectionTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Owlv2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Test_initialization is tested in individual model tests")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Test_forward_signature is tested in individual model tests")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="OwlV2 does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init).to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # OWLV2 needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            loaded_model = loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict:
                if key not in model_state_dict:
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlv2-base-patch16-ensemble"
        model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class Owlv2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "google/owlv2-base-patch16"
        model = Owlv2Model.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )
        expected_logits = torch.tensor([[-6.2229, -8.2601]], device=torch_device)
        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        model_name = "google/owlv2-base-patch16"
        model = Owlv2Model.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        processor.image_processor.size = {"height": 1024, "width": 1024}

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )
        expected_logits = torch.tensor([[-6.2520, -8.2970]], device=torch_device)
        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)
        expected_shape = torch.Size((1, 4097, 768))
        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)

        # Owlv2ForObjectDetection part.
        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)
        processor.image_processor.size = {"height": 1024, "width": 1024}

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        num_queries = int((inputs.pixel_values.shape[-1] / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_boxes = torch.tensor(
            [[0.2407, 0.0553, 0.4636], [0.1082, 0.0494, 0.1861], [0.2459, 0.0527, 0.4398]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs, interpolate_pos_encoding=True)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int((inputs.pixel_values.shape[-1] / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

        # Deactivate interpolate_pos_encoding on same model, and use default image size.
        # Verify the dynamic change caused by the activation/deactivation of interpolate_pos_encoding of variables: self.sqrt_num_patches, self.box_bias from (OwlViTForObjectDetection).
        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=False)

        num_queries = int((inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_default_box_bias = torch.tensor(
            [
                [-4.0717, -4.0717, -4.0717, -4.0717],
                [-3.3644, -4.0717, -4.0717, -4.0717],
                [-2.9425, -4.0717, -4.0717, -4.0717],
            ]
        )

        torch.testing.assert_close(model.box_bias[:3, :4], expected_default_box_bias, rtol=1e-4, atol=1e-4)

        # Interpolate with any resolution size.
        processor.image_processor.size = {"height": 1264, "width": 1024}

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        num_queries = int(
            (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
            * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
        )
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_boxes = torch.tensor(
            [[0.2438, 0.0945, 0.4675], [0.1361, 0.0431, 0.2406], [0.2465, 0.0428, 0.4429]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs, interpolate_pos_encoding=True)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int(
            (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
            * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
        )
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

    @slow
    def test_inference_object_detection(self):
        model_name = "google/owlv2-base-patch16"
        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        text_labels = [["a photo of a cat", "a photo of a dog"]]
        inputs = processor(
            text=text_labels,
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_slice_logits = torch.tensor(
            [[-21.413497, -21.612638], [-19.008193, -19.548841], [-20.958896, -21.382694]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_slice_logits, rtol=1e-4, atol=1e-4)
        expected_slice_boxes = torch.tensor(
            [[0.241309, 0.051896, 0.453267], [0.139474, 0.045701, 0.250660], [0.233022, 0.050479, 0.427671]],
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)
        resulted_slice_boxes = outputs.pred_boxes[0, :3, :3]
        max_diff = torch.max(torch.abs(resulted_slice_boxes - expected_slice_boxes)).item()
        self.assertLess(max_diff, 3e-4)

        # test post-processing
        post_processed_output = processor.post_process_grounded_object_detection(outputs)
        self.assertIsNone(post_processed_output[0]["text_labels"])

        post_processed_output_with_text_labels = processor.post_process_grounded_object_detection(
            outputs, text_labels=text_labels
        )

        objects_labels = post_processed_output_with_text_labels[0]["labels"].tolist()
        self.assertListEqual(objects_labels, [0, 0])

        objects_text_labels = post_processed_output_with_text_labels[0]["text_labels"]
        self.assertIsNotNone(objects_text_labels)
        self.assertListEqual(objects_text_labels, ["a photo of a cat", "a photo of a cat"])

    @slow
    def test_inference_one_shot_object_detection(self):
        model_name = "google/owlv2-base-patch16"
        model = Owlv2ForObjectDetection.from_pretrained(model_name).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_slice_boxes = torch.tensor(
            [[0.2413, 0.0519, 0.4533], [0.1395, 0.0457, 0.2507], [0.2330, 0.0505, 0.4277]],
        ).to(torch_device)
        torch.testing.assert_close(outputs.target_pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_one_shot_object_detection_fp16(self):
        model_name = "google/owlv2-base-patch16"
        model = Owlv2ForObjectDetection.from_pretrained(model_name, torch_dtype=torch.float16).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))
