import pytest
import litellm.gpt_cache

import fastrepl


@pytest.fixture
def mock_completion(monkeypatch):
    def ret(return_value):
        def mock(*args, **kwargs):
            return {
                "choices": [
                    {"message": {"content": return_value}, "finish_reason": "stop"}
                ]
            }

        monkeypatch.setattr(litellm.gpt_cache, "completion", mock)

    return ret


class TestLLMClassificationHead:
    @pytest.mark.parametrize(
        "return_value, labels",
        [
            (
                "B",
                {
                    "A": "this is A",
                    "B": "this is B",
                    "C": "this is C",
                },
            )
        ],
    )
    def test_return_result(self, mock_completion, return_value, labels):
        mock_completion(return_value)

        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels=labels,
        )
        assert eval.compute("") == return_value

    @pytest.mark.parametrize(
        "return_value, labels",
        [
            (
                "D",
                {
                    "A": "this is A",
                    "B": "this is B",
                    "C": "this is C",
                },
            )
        ],
    )
    def test_return_none(self, mock_completion, return_value, labels):
        mock_completion(return_value)

        eval = fastrepl.LLMClassificationHead(
            context="test",
            labels=labels,
        )
        assert eval.compute("") is None


class TestLLMGradingHead:
    @pytest.mark.parametrize(
        "return_value, number_from, number_to",
        [
            ("2", 1, 3),
            ("3", 1, 5),
        ],
    )
    def test_return_result(self, mock_completion, return_value, number_from, number_to):
        mock_completion(return_value)

        eval = fastrepl.LLMGradingHead(
            context="test",
            number_from=number_from,
            number_to=number_to,
        )
        assert eval.compute("") == return_value

    @pytest.mark.parametrize(
        "return_value, number_from, number_to",
        [
            ("0", 1, 3),
            ("6", 1, 5),
        ],
    )
    def test_return_none(self, mock_completion, return_value, number_from, number_to):
        mock_completion(return_value)

        eval = fastrepl.LLMGradingHead(
            context="test",
            number_from=number_from,
            number_to=number_to,
        )
        assert eval.compute("") is None
