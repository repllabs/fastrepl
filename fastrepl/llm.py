from typing import List, Dict, Any
import functools
import traceback

import backoff
import openai.error

from fastrepl.warnings import (
    warn,
    CompletionTruncatedWarning,
    UnknownLLMExceptionWarning,
)
from fastrepl.errors import TokenizeNotImplementedError
from fastrepl.utils import getenv, debug


import litellm
import litellm.exceptions
from litellm.caching import Cache


def custom_get_cache_key(*args, **kwargs):  # pragma: no cover
    model = str(kwargs.get("model", ""))
    messages = str(kwargs.get("messages", ""))
    temperature = str(kwargs.get("temperature", ""))
    logit_bias = str(kwargs.get("logit_bias", ""))

    key = f"{model}/{messages}/{temperature}/{logit_bias}"
    return key


litellm.telemetry = False  # pragma: no cover
litellm.cache = Cache()  # pragma: no cover
litellm.cache.get_cache_key = custom_get_cache_key  # pragma: no cover

litellm_config = {
    "function": "completion",
    "model": {
        "gpt-3.5-turbo": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-3.5-turbo-16k"}
            }
        },
        "gpt-3.5-turbo-0301": {
            "error_handling": {
                "ContextWindowExceededError": {
                    "fallback_model": "gpt-3.5-turbo-16k-0301"
                }
            }
        },
        "gpt-3.5-turbo-0613": {
            "error_handling": {
                "ContextWindowExceededError": {
                    "fallback_model": "gpt-3.5-turbo-16k-0613"
                }
            }
        },
        "gpt-4": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-4-32k"}
            }
        },
        "gpt-4-0314": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-4-32k-0314"}
            }
        },
        "gpt-4-0613": {
            "error_handling": {
                "ContextWindowExceededError": {"fallback_model": "gpt-4-32k-0613"}
            }
        },
    },
}


class RetryConstantException(Exception):
    pass


class RetryExpoException(Exception):
    pass


def handle_llm_exception(e: Exception):
    if isinstance(
        e,
        (
            openai.error.APIError,
            openai.error.TryAgain,
            openai.error.Timeout,
            openai.error.ServiceUnavailableError,
        ),
    ):
        raise RetryConstantException from e
    elif isinstance(e, openai.error.RateLimitError):
        raise RetryExpoException from e
    elif isinstance(
        e,
        (
            openai.error.APIConnectionError,
            openai.error.InvalidRequestError,
            openai.error.AuthenticationError,
            openai.error.PermissionError,
            openai.error.InvalidAPIType,
            openai.error.SignatureVerificationError,
        ),
    ):
        raise e
    else:
        name = type(e).__name__
        trace = traceback.format_exc()
        warn(UnknownLLMExceptionWarning, context=f"{name}: {trace}")
        raise e


@backoff.on_exception(
    wait_gen=backoff.constant,
    exception=(RetryConstantException),
    raise_on_giveup=True,
    max_tries=3,
    interval=3,
)
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(RetryExpoException),
    raise_on_giveup=True,
    jitter=backoff.full_jitter,
    max_value=100,
    factor=1.5,
)
def completion(  # pragma: no cover
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    logit_bias: Dict[int, int] = {},
    max_tokens: int = 200,
) -> Dict[str, Any]:
    """
    https://docs.litellm.ai/docs/providers
    """
    try:
        result = litellm.completion_with_config(
            litellm_config,
            model=model,
            messages=messages,
            temperature=temperature,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
        )
        content = result["choices"][0]["message"]["content"]

        if result["choices"][0]["finish_reason"] == "length":
            warn(CompletionTruncatedWarning, context=content)

        # TODO: debug call should be done in eval side
        debug({"llm_input": messages, "llm_output": content})

        return result
    except Exception as e:
        handle_llm_exception(e)

    raise Exception  # to make mypy happy


@functools.lru_cache(maxsize=None)
def tokenize(model: str, text: str) -> List[int]:
    if model.startswith("command"):
        import cohere

        co = cohere.Client(getenv("COHERE_API_KEY", ""))
        response = co.tokenize(text=text, model="command")
        return response.tokens
    elif model.startswith("gpt"):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)

    raise TokenizeNotImplementedError(model)
