# AGENTS.md Guide for Hugging Face Transformers

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/sr
Many models in the codebase have similar code, but it is not shared by inheritance because we want each model file to be self-contained.
We use two mechanisms to keep this code in sync:

- "Copied from" syntax. Functions or entire classes can have a comment at the top like this: `# Copied from transformers.models.llama.modeling_llama.rotate_half` or `# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5`
  These comments are actively checked by the style tools, and copies will automatically be updated when the base code is updated. If you need to update a copied function, you should
  either update the base function and use `make fixup` to propagate the change to all copies, or simply remove the `# Copied from` comment if that is inappropriate.
- "Modular" files. These files briefly define models by composing them using inheritance from other models. They are not meant to be used directly. Instead, the style tools
  automatically generate a complete modeling file, like `modeling_bert.py`, from the modular file like `modular_bert.py`. If a model has a modular file, the modeling file
  should never be edited directly! Instead, changes should be made in the modular file, and then you should run `make fixup` to update the modeling file automatically.

When adding new models, you should prefer `modular` style.

## Testing

After making changes, you should usually run `make fixup` to ensure any copies and modular files are updated, and then test all affected models. This includes both
the model you made the changes in and any other models that were updated by `make fixup`. Tests can be run with `pytest tests/models/[name]/test_modeling_[name].py`
If your changes affect code in other classes like tokenizers or processors, you should run those tests instead, like `test_processing_[name].py` or `test_tokenization_[name].py`.

In order to run tests, you may need to install dependencies. You can do this with `pip install -e .[testing]`. You will probably also need to `pip install torch accelerate` if your environment does not already have them.
