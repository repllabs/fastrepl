import pytest

import io
import fastrepl

from rich.console import Console


class TestHumanEvalWithRich:
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
    def test_basic(self, input, expected):
        console = Console(file=io.StringIO())

        eval = fastrepl.HumanClassifierRich(
            labels={"a": "this is a", "b": "this is b"},
            console=console,
            stream=io.StringIO(input),
        )

        assert eval.compute("some sample") == expected
