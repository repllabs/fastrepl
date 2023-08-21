import functools
from typing import Dict

from fastrepl.run import SUPPORTED_MODELS, tokenize


@functools.lru_cache(maxsize=None)
def logit_bias_for_classification(model: SUPPORTED_MODELS, keys: str) -> Dict[int, int]:
    if len(keys) != len(set(keys)):
        raise ValueError("all characters in keys must be unique")

    if model == "command-nightly":
        COHERE_MAX = 10
        return {tokenize(model, k)[0]: COHERE_MAX for k in keys}
    elif model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]:
        OPENAI_MAX = 100
        return {tokenize(model, k)[0]: OPENAI_MAX for k in keys}
    else:
        return {}
