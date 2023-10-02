import pytest

import fastrepl.runner
from fastrepl.dataset import Dataset


@pytest.fixture
def mock_runs(monkeypatch):
    def ret(values):
        iter_values = iter(values)

        def mock_run(*args, **kwargs):
            return next(iter_values)

        monkeypatch.setattr(fastrepl.runner.LocalEvaluatorRunner, "_run", mock_run)

    return ret


class TestLocalRunner:
    def test_num_1(self, mock_runs):
        mock_runs([[1]])

        ds = Dataset.from_dict({"sample": [1]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.local_runner(evaluator=eval, dataset=ds).run(num=1)

        assert result.column_names == ["sample", "result"]

    def test_num_2(self, mock_runs):
        mock_runs([[1, 2, 3, 4], [1, 2, 3, 5]])

        ds = Dataset.from_dict({"sample": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.local_runner(evaluator=eval, dataset=ds).run(num=2)

        assert result.column_names == ["sample", "result"]
        assert result["result"] == [[1, 1], [2, 2], [3, 3], [4, 5]]

    def test_num_2_handle_none(self, mock_runs):
        mock_runs([[1, 2, 3, 4], [1, 2, 3, None]])

        ds = Dataset.from_dict({"sample": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        result = fastrepl.local_runner(evaluator=eval, dataset=ds).run(num=2)

        assert result.column_names == ["sample", "result"]
        assert result["result"] == [[1, 1], [2, 2], [3, 3], [4, None]]

    def test_validation(self):
        ds = Dataset.from_dict({"input": [1, 2, 3, 4]})
        eval = fastrepl.SimpleEvaluator(
            node=fastrepl.LLMClassificationHead(context="", labels={})
        )

        with pytest.raises(ValueError):
            fastrepl.local_runner(evaluator=eval, dataset=ds)


class TestCustomRunner:
    def test_args(self):
        def adder(x, y):
            return x + y

        r = fastrepl.runner.LocalCustomRunner(adder)
        result = r.run(args_list=[(1, 2), (2, 3), (3, 4)])
        assert result.to_dict() == {"sample": [3, 5, 7]}

    def test_kwds(self):
        def adder(*, x, y):
            return x + y

        r = fastrepl.runner.LocalCustomRunner(adder)
        result = r.run(kwds_list=[{"x": 1, "y": 2}, {"x": 2, "y": 3}])
        assert result.to_dict() == {"sample": [3, 5]}

    def test_args_kwds(self):
        def adder(x, *, y):
            return x + y

        r = fastrepl.runner.LocalCustomRunner(adder)
        result = r.run(args_list=[(1,), (2,)], kwds_list=[{"y": 2}, {"y": 3}])
        assert result.to_dict() == {"sample": [3, 5]}
