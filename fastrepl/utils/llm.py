import traceback

import openai.error

from fastrepl.warnings import (
    warn,
    UnknownLLMExceptionWarning,
)


class RetryConstantException(Exception):
    pass


class RetryExpoException(Exception):
    pass


def raise_openai_exception_for_retry(e: Exception):
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
