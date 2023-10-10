from typing import Literal, Optional, List

import backoff
import openai.error
from wrapt_timeout_decorator import timeout

from fastrepl.utils import (
    raise_openai_exception_for_retry,
    RetryConstantException,
    RetryExpoException,
)
from lazy_imports import try_import

with try_import() as optional_package_import:
    from datasets import Dataset
    from langchain.chat_models import ChatOpenAI

    from ragas.metrics.base import MetricWithLLM, EvaluationMode
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.metrics.critique import (
        harmfulness,
        maliciousness,
        coherence,
        correctness,
        conciseness,
    )

    from ragas import evaluate as _evaluate
    from fastrepl.utils import suppress

    @timeout(15, timeout_exception=openai.error.Timeout)
    @suppress
    def evaluate(dataset: Dataset, metric: MetricWithLLM) -> Optional[float]:
        result = _evaluate(dataset=dataset, metrics=[metric])
        return list(result.scores[0].values())[0]


RAGAS_METRICS = Literal[  # pragma: no cover
    "Faithfulness",
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "harmfulness",
    "maliciousness",
    "coherence",
    "correctness",
    "conciseness",
]

from fastrepl.eval.base import BaseRAGEvalNode


class RAGAS(BaseRAGEvalNode):
    def __init__(
        self,
        metric: RAGAS_METRICS,
        model: Literal[  # TODO
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
        ] = "gpt-3.5-turbo",
    ):
        optional_package_import.check()

        self._metric = self._load_metric(model, metric)

    def _load_metric(
        self,
        model_name: str,
        metric_name: RAGAS_METRICS,
    ) -> MetricWithLLM:
        if not model_name.startswith("gpt"):
            raise NotImplementedError

        llm = ChatOpenAI(model=model_name)  # type: ignore[call-arg]

        metric: MetricWithLLM
        if metric_name == "AnswerRelevancy":
            metric = AnswerRelevancy(llm=llm, batch_size=1)
        elif metric_name == "ContextRecall":
            metric = ContextRecall(llm=llm, batch_size=1)
        elif metric_name == "ContextPrecision":
            metric = ContextPrecision(llm=llm, batch_size=1)
        elif metric_name == "Faithfulness":
            metric = Faithfulness(llm=llm, batch_size=1)
        elif metric_name == "harmfulness":
            harmfulness.llm = llm
            metric = harmfulness
        elif metric_name == "maliciousness":
            maliciousness.llm = llm
            metric = maliciousness
        elif metric_name == "coherence":
            coherence.llm = llm
            metric = coherence
        elif metric_name == "correctness":
            correctness.llm = llm
            metric = correctness
        elif metric_name == "conciseness":
            conciseness.llm = llm
            metric = conciseness
        else:
            raise ValueError

        return metric

    def inputs(self) -> List[str]:
        if self._metric.evaluation_mode == EvaluationMode.qac:
            return ["question", "answer", "contexts"]
        elif self._metric.evaluation_mode == EvaluationMode.qa:
            return ["question", "answer"]
        elif self._metric.evaluation_mode == EvaluationMode.qc:
            return ["question", "contexts"]
        elif self._metric.evaluation_mode == EvaluationMode.gc:
            return ["ground_truths", "contexts"]
        else:
            raise ValueError

    def run(
        self,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Optional[float]:
        ds: Dataset

        if self._metric.evaluation_mode == EvaluationMode.qac:
            if question is None or answer is None or contexts is None:
                raise ValueError
            if len(contexts) == 0:
                raise ValueError

            ds = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                }
            )
        elif self._metric.evaluation_mode == EvaluationMode.qa:
            if question is None or answer is None:
                raise ValueError

            ds = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                }
            )
        elif self._metric.evaluation_mode == EvaluationMode.qc:
            if question is None:
                raise ValueError
            if contexts is None or len(contexts) == 0:
                raise ValueError

            ds = Dataset.from_dict(
                {
                    "question": [question],
                    "contexts": [contexts],
                }
            )
        elif self._metric.evaluation_mode == EvaluationMode.gc:
            if contexts is None or ground_truths is None:
                raise ValueError
            if len(contexts) != len(ground_truths) or len(contexts) == 0:
                raise ValueError

            ds = Dataset.from_dict(
                {
                    "ground_truths": [ground_truths],
                    "contexts": [contexts],
                }
            )
        else:
            raise ValueError

        return self._evaluate_with_retry(dataset=ds, metric=self._metric)

    @backoff.on_exception(
        wait_gen=backoff.constant,
        exception=(RetryConstantException),
        raise_on_giveup=True,
        max_tries=3,
        interval=3,
    )
    @backoff.on_exception(
        wait_gen=backoff.expo,
        exception=(RetryExpoException),
        raise_on_giveup=True,
        jitter=backoff.full_jitter,
        max_value=100,
        factor=1.5,
    )
    def _evaluate_with_retry(
        self, dataset: Dataset, metric: MetricWithLLM
    ) -> Optional[float]:
        try:
            return evaluate(dataset=dataset, metric=metric)
        except Exception as e:
            raise_openai_exception_for_retry(e)

        raise Exception  # to make mypy happy
