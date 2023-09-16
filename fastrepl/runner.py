from abc import ABC, abstractmethod
from typing import Optional, List, Any
import inspect

from multiprocessing.pool import ThreadPool
from datasets import Dataset
from rich.progress import Progress, TaskID

import fastrepl
from fastrepl.utils import getenv

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
        output_feature="result",
        output_feature_multiple="results",
    ) -> None:
        self._evaluator = evaluator
        self._dataset = dataset

        self._output_feature = output_feature
        self._output_feature_multiple = output_feature_multiple

        self._input_features = [
            param for param in inspect.signature(evaluator.run).parameters.keys()
        ]

    def _run_eval(self, **kwargs) -> Optional[Any]:
        return self._evaluator.run(**kwargs)

    def _run(self, progress: Progress, task_id: TaskID) -> List[Optional[Any]]:
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

    def run(self, num=1) -> Dataset:
        with Progress() as progress:
            msg = "[cyan]Processing..."
            task_id = progress.add_task(msg, total=len(self._dataset) * num)

            if num == 1:
                result = self._run(progress, task_id)

                return self._dataset.add_column(
                    self._output_feature,
                    result,
                )
            else:
                results = [self._run(progress, task_id) for _ in range(num)]

                return self._dataset.add_column(
                    self._output_feature_multiple,
                    list(zip(*results)),
                )


class LocalRunnerREPL(LocalRunner):
    pass


class RemoteRunner(BaseRunner):
    def __init__(self) -> None:
        raise NotImplementedError
