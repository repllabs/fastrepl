import pytest
from datasets import Dataset

import fastrepl


@pytest.fixture
def mock_runs(monkeypatch):
    def ret(values):
        iter_values = iter(values)

        def mock_run(*args, **kwargs):
            return next(iter_values)

        monkeypatch.setattr(fastrepl.LocalRunner, "_run", mock_run)

    return ret


class TestLocalRunner:
    def test_num_1(self, mock_runs):
        mock_runs([[1]])

        ds = Dataset.from_dict({"sample": [1]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run(num=1)

        assert result.column_names == ["sample", "result"]

    def test_num_2(self, mock_runs):
        mock_runs([[1, 2, 3, 4], [1, 2, 3, 5]])

        ds = Dataset.from_dict({"sample": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run(num=2)

        assert result.column_names == ["sample", "result"]
        assert result["result"] == [[1, 1], [2, 2], [3, 3], [4, 5]]

    def test_num_2_handle_none(self, mock_runs):
        mock_runs([[1, 2, 3, 4], [1, 2, 3, None]])

        ds = Dataset.from_dict({"sample": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.LocalRunner(evaluator=eval, dataset=ds).run(num=2)

        assert result.column_names == ["sample", "result"]
        assert result["result"] == [[1, 1], [2, 2], [3, 3], [4, None]]

    def test_validation(self):
        ds = Dataset.from_dict({"input": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        with pytest.raises(ValueError):
            fastrepl.LocalRunner(evaluator=eval, dataset=ds)
