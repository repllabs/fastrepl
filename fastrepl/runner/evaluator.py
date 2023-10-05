from typing import Optional, Callable, List, Any

from multiprocessing.pool import ThreadPool
from rich.progress import Progress

import fastrepl
from fastrepl.dataset import Dataset
from fastrepl.utils import getenv, console
from fastrepl.runner.base import BaseRunner

NUM_THREADS = getenv("NUM_THREADS", 8)


class LocalEvaluatorRunner(BaseRunner):
    def __init__(
        self,
        evaluator: fastrepl.Evaluator,
        dataset: Dataset,
        output_feature="result",
    ) -> None:
        self._input_features = evaluator.inputs()
        self._output_feature = output_feature

        if any(feature not in dataset.column_names for feature in self._input_features):
            eval_name = type(evaluator).__name__

            raise ValueError(  # TODO: custom error
                f"{eval_name} requires {self._input_features!r}, but the provided dataset has {dataset.column_names!r}"
            )

        self._evaluator = evaluator
        self._dataset = dataset

    def _run_single(self, cb: Callable[[], None]) -> List[Optional[Any]]:
        results = []

        with ThreadPool(min(NUM_THREADS, len(self._dataset))) as pool:
            futures = [
                pool.apply_async(
                    self._evaluator.run,
                    kwds={
                        feature: value
                        for feature, value in zip(self._input_features, values)
                    },
                )
                for values in zip(
                    *[self._dataset[feature] for feature in self._input_features]
                )
            ]

            for future in futures:
                results.append(future.get())
                cb()

        return results

    def run(self, num=1, show_progress=True) -> Dataset:
        disable = not show_progress

        try:
            with Progress(console=console, disable=disable) as progress:
                msg = "[cyan]Processing..."
                task_id = progress.add_task(msg, total=len(self._dataset) * num)
                cb = lambda: progress.update(task_id, advance=1, refresh=True)

                if num > 1:
                    results = [self._run_single(cb) for _ in range(num)]
                    multiple_data = [list(item) for item in zip(*results)]
                    return self._dataset.add_column(self._output_feature, multiple_data)

                single_data = self._run_single(cb)
                return self._dataset.add_column(self._output_feature, single_data)
        except ValueError as e:
            if "I/O operation on closed file" in str(e):
                console.print("[cyan]Please re-run with `show_progress=False`")
            else:
                raise e

        return Dataset.from_dict({})


class RemoteEvaluatorRunner(LocalEvaluatorRunner):
    def __init__(
        self,
        evaluator: fastrepl.Evaluator,
        dataset: Dataset,
        output_feature="result",
    ) -> None:
        raise NotImplementedError
