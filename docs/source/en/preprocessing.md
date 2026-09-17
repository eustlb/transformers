<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Preprocessing

Preprocessors convert raw inputs such as images, videos, and audio into the tensors a model expects. Each modality
has its own transformations, but preprocessors share a common configuration API through `PreprocessingMixin`.

- `from_pretrained` loads a preprocessor configuration from the Hugging Face Hub or a local
  directory.
- `save_pretrained` saves the configuration so it can be loaded again.
- `to_dict` and `to_json_file` serialize the configuration.
- Calling a preprocessor validates its supported options, fills omitted values from the saved configuration, and
  returns a [`BatchFeature`]. Call-time options apply to that call without changing the saved configuration.

Refer to the modality guide for accepted inputs, available transformations, and examples:

- [Image processors](./image_processors)
- [Video processors](./video_processors)
- [Audio processors](./audio_processors)
