import pytest

import io
import fastrepl

from rich.console import Console


@pytest.mark.parametrize(
    "input, expected",
    [
        ("a", "a"),
        ("a\n", "a"),
        ("c\na", "a"),
        ("c\na\n", "a"),
        ("A\nc\na", "a"),
    ],
)
def test_human_eval_with_rich(input, expected):
    console = Console(file=io.StringIO())

    eval = fastrepl.HumanClassifierRich(
        labels={"a": "this is a", "b": "this is b"},
        console=console,
        stream=io.StringIO(input),
    )

    actual = eval.compute(sample="some sample")
    assert actual == expected
