import pytest
from datasets import Dataset

from fastrepl.loop import Evaluator


class TestEvaluator:
    def test_no_input(self):
        ds = Dataset.from_dict({"a": [1]})

        with pytest.raises(ValueError):
            Evaluator(
                dataset=ds,
                evals=[],
            )

    def test_features(self):
        input_ds = Dataset.from_dict({"input": [1]})
        output_ds = Evaluator(
            dataset=input_ds,
            evals=[],
        ).run()

        assert len(output_ds.features) == 2
        assert "input" in output_ds.features
        assert "output" in output_ds.features
