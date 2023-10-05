from typing import Union, Callable, overload

from fastrepl.dataset import Dataset

import fastrepl
from fastrepl.runner.evaluator import LocalEvaluatorRunner, RemoteEvaluatorRunner
from fastrepl.runner.generator import LocalGeneratorRunner, RemoteGeneratorRunner
from fastrepl.runner.others import LocalCustomRunner


@overload
def local_runner(*, fn: Callable, output_feature: str) -> LocalCustomRunner:
    ...


@overload
def local_runner(*, generator: fastrepl.Generator) -> LocalGeneratorRunner:
    ...


@overload
def local_runner(
    *, evaluator: fastrepl.Evaluator, dataset: Dataset, output_feature: str
) -> LocalEvaluatorRunner:
    ...


def local_runner(
    **kwargs,
) -> Union[LocalGeneratorRunner, LocalEvaluatorRunner, LocalCustomRunner]:
    if "generator" in kwargs:
        return LocalGeneratorRunner(generator=kwargs["generator"])

    if "evaluator" in kwargs:
        return LocalEvaluatorRunner(
            evaluator=kwargs["evaluator"],
            dataset=kwargs["dataset"],
            output_feature=kwargs.get("output_feature", "result"),
        )

    if "fn" in kwargs:
        return LocalCustomRunner(
            fn=kwargs["fn"],
            output_feature=kwargs.get("output_feature", "result"),
        )

    raise ValueError


@overload
def remote_runner(*, generator: fastrepl.Generator) -> RemoteGeneratorRunner:
    return RemoteGeneratorRunner(generator)


@overload
def remote_runner(
    *, evaluator: fastrepl.Evaluator, dataset: Dataset, output_feature="result"
) -> RemoteEvaluatorRunner:
    return RemoteEvaluatorRunner(evaluator, dataset, output_feature)


def remote_runner(**kwargs) -> Union[RemoteEvaluatorRunner, RemoteGeneratorRunner]:
    if "generator" in kwargs:
        return RemoteGeneratorRunner(generator=kwargs["generator"])

    return RemoteEvaluatorRunner(**kwargs)


__all__ = ["local_runner", "remote_runner"]
