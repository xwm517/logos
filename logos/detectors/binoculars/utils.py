"""From https://github.com/ahans30/Binoculars (binoculars/utils.py).

MODIFICATION: the authors' error message calls .name_of_path on the two
arguments, which are strings, so the failure path raises AttributeError rather
than the intended ValueError. The IDs are used directly here instead.
"""

from transformers import AutoTokenizer


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1).vocab
        == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(
            f"Tokenizers are not identical for {model_id_1} and {model_id_2}."
        )