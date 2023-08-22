import functools
from typing import List

from rich.progress import Progress
from datasets import Dataset

from fastrepl.eval.model.base import BaseModelEval


class Evaluator:
    __slots__ = ["dataset", "evals"]

    def __init__(self, dataset: Dataset, evals: List[BaseModelEval]) -> None:
        if "input" not in dataset.features:
            raise ValueError("Dataset must have input column.")

        self.dataset = dataset
        self.evals = evals

    def _run_evals(self, input: str, context="") -> str:
        return functools.reduce(
            lambda previous, eval: eval.compute(input, previous), self.evals, context
        )

    def run(self) -> Dataset:
        results = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(self.dataset))

            for row in self.dataset:
                result = self._run_evals(row["input"])
                results.append(result)
                progress.update(task, advance=1)

        return self.dataset.add_column("output", results)
