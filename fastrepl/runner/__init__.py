from typing import Union, overload

from datasets import Dataset

import fastrepl
from fastrepl.runner.evaluator import LocalEvaluatorRunner, RemoteEvaluatorRunner
from fastrepl.runner.generator import LocalGeneratorRunner, RemoteGeneratorRunner


@overload
def local_runner(*, generator: fastrepl.Generator) -> LocalGeneratorRunner:
    return LocalGeneratorRunner(generator)


@overload
def local_runner(
    *, evaluator: fastrepl.Evaluator, dataset: Dataset, output_feature="result"
) -> LocalEvaluatorRunner:
    return LocalEvaluatorRunner(evaluator, dataset, output_feature)


def local_runner(**kwargs) -> Union[LocalEvaluatorRunner, LocalGeneratorRunner]:
    if "generator" in kwargs:
        return LocalGeneratorRunner(generator=kwargs["generator"])

    return LocalEvaluatorRunner(**kwargs)


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
