from abc import ABC, abstractmethod
from typing import Optional, List
import inspect

from multiprocessing.pool import ThreadPool
from datasets import Dataset
from rich.progress import Progress, TaskID

import fastrepl
from fastrepl.utils import getenv, kappa
from fastrepl.warnings import warn, InconsistentPredictionWarning

NUM_THREADS = getenv("NUM_THREADS", 8)


class BaseRunner(ABC):
    @abstractmethod
    def run(self) -> Dataset:
        pass


class LocalRunner(BaseRunner):
    def __init__(
        self,
        evaluator: fastrepl.Evaluator,
        dataset: Dataset,
    ) -> None:
        self._evaluator = evaluator
        self._dataset = dataset

        self._input_features = [
            param for param in inspect.signature(evaluator.run).parameters.keys()
        ]

    def _run_eval(self, **kwargs) -> Optional[str]:
        return self._evaluator.run(**kwargs)  # TODO

    def _run(self, progress: Progress, task_id: TaskID) -> List[Optional[str]]:
        results = []

        with ThreadPool(NUM_THREADS) as pool:
            futures = [
                pool.apply_async(
                    self._run_eval,
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
                progress.update(task_id, advance=1, refresh=True)
        return results

    def run(self) -> Dataset:
        with Progress() as progress:
            task_id = progress.add_task(
                "[cyan]Processing...",
                total=len(self._dataset),
            )

            self._run(progress, task_id)


class LocalRunnerREPL(LocalRunner):
    pass


class RemoteRunner(BaseRunner):
    def __init__(self) -> None:
        raise NotImplementedError
