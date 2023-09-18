import pytest

import fastrepl
from fastrepl.eval.base import BaseEvalNode


class MockEval(BaseEvalNode):
    def compute(self, sample: str) -> str:
        return sample + "0"


@pytest.mark.parametrize(
    "node, sample, result",
    [
        (MockEval(), "1", "10"),
        (MockEval(), "12", "120"),
    ],
)
def test_simple_evaluator(node, sample, result):
    assert fastrepl.SimpleEvaluator(node).run(sample=sample) == result
