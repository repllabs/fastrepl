import pytest
import openai.error
import litellm

from fastrepl.llm import (
    raise_openai_exception_for_retry,
    RetryConstantException,
    RetryExpoException,
    completion,
)


class TestCompletion:
    def test_kwargs(self):
        with pytest.raises(TypeError):
            completion("gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}])


class TestHandleLLMException:
    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.APIError(),
            openai.error.TryAgain(),
            openai.error.Timeout(),
            openai.error.ServiceUnavailableError(),
        ],
    )
    def test_constant(self, exception):
        with pytest.raises(RetryConstantException):
            raise_openai_exception_for_retry(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.RateLimitError(),
        ],
    )
    def test_expo(self, exception):
        with pytest.raises(RetryExpoException):
            raise_openai_exception_for_retry(exception)

    @pytest.mark.parametrize(
        "exception",
        [
            openai.error.APIConnectionError(""),
            openai.error.InvalidRequestError("", ""),
            openai.error.AuthenticationError(),
            openai.error.PermissionError(),
            openai.error.InvalidAPIType(),
            openai.error.SignatureVerificationError("", ""),
        ],
    )
    def test_no_retry(self, exception):
        with pytest.raises(type(exception)):
            raise_openai_exception_for_retry(exception)


class TestContextFallback:
    def test_short(self, monkeypatch):
        def mock(**kwargs):
            if kwargs.get("model") == "gpt-3.5-turbo":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            elif kwargs.get("model") == "gpt-3.5-turbo-16k":
                return {
                    "choices": [{"finish_reason": "stop", "message": {"content": ""}}]
                }
            else:
                raise NotImplementedError

        monkeypatch.setattr(litellm, "completion", mock)

        completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "8k tokens"}],
        )

    def test_long(self, monkeypatch):
        def mock(**kwargs):
            if kwargs.get("model") == "gpt-3.5-turbo":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            elif kwargs.get("model") == "gpt-3.5-turbo-16k":
                raise litellm.exceptions.ContextWindowExceededError("", "", "")
            else:
                raise NotImplementedError

        monkeypatch.setattr(litellm, "completion", mock)

        with pytest.raises(litellm.exceptions.ContextWindowExceededError):
            completion(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": "24k tokens"}],
            )
