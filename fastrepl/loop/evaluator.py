import time
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

    def run(self) -> Dataset:
        result = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(self.dataset))
            for row in self.dataset:
                time.sleep(0.01)
                progress.update(task, advance=1)
                result.append(0)
        return self.dataset.add_column("output", result)
