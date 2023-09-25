from typing import Literal, Optional, List

from datasets import Dataset
from langchain.chat_models import ChatOpenAI

from lazy_imports import try_import

with try_import() as optional_package_import:
    from ragas import evaluate
    from ragas.metrics.base import MetricWithLLM, EvaluationMode
    from ragas.metrics import (
        AnswerRelevancy,
        ContextRecall,
        ContextRelevancy,
        Faithfulness,
    )
    from ragas.metrics.critique import (
        harmfulness,
        maliciousness,
        coherence,
        correctness,
        conciseness,
    )


RAGAS_METRICS = Literal[  # pragma: no cover
    "AnswerRelevancy",
    "ContextRecall",
    "ContextRelevancy",
    "Faithfulness",
    "harmfulness",
    "maliciousness",
    "coherence",
    "correctness",
    "conciseness",
]

from fastrepl.llm import SUPPORTED_MODELS
from fastrepl.utils import suppress
from fastrepl.eval.base import RAGEvalNode


class RAGAS(RAGEvalNode):
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

        self.metric = self._load_metric(model, metric)

    def _load_metric(
        self,
        model_name: SUPPORTED_MODELS,
        metric_name: RAGAS_METRICS,
    ) -> MetricWithLLM:
        if not model_name.startswith("gpt"):
            raise NotImplementedError

        llm = ChatOpenAI(model=str(model_name))  # type: ignore[call-arg]

        metric: MetricWithLLM
        if metric_name == "AnswerRelevancy":
            metric = AnswerRelevancy(llm=llm, batch_size=1)
        elif metric_name == "ContextRecall":
            metric = ContextRecall(llm=llm, batch_size=1)
        elif metric_name == "ContextRelevancy":
            metric = ContextRelevancy(llm=llm, batch_size=1)
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

        # TODO: https://github.com/explodinggradients/ragas/pull/118
        metric.strictness = 1
        return metric

    def inputs(self) -> List[str]:
        if self.metric.evaluation_mode == EvaluationMode.qac:
            return ["question", "answer", "contexts"]
        elif self.metric.evaluation_mode == EvaluationMode.qa:
            return ["question", "answer"]
        elif self.metric.evaluation_mode == EvaluationMode.qc:
            return ["question", "contexts"]
        elif self.metric.evaluation_mode == EvaluationMode.gc:
            return ["ground_truths", "contexts"]
        else:
            raise ValueError

    def compute(
        self,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Optional[float]:
        ds: Dataset

        if self.metric.evaluation_mode == EvaluationMode.qac:
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
        elif self.metric.evaluation_mode == EvaluationMode.qa:
            if question is None or answer is None:
                raise ValueError

            ds = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                }
            )
        elif self.metric.evaluation_mode == EvaluationMode.qc:
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
        elif self.metric.evaluation_mode == EvaluationMode.gc:
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

        return self._evaluate(ds, self.metric)

    @suppress
    def _evaluate(self, ds: Dataset, metric: MetricWithLLM) -> Optional[float]:
        result = evaluate(dataset=ds, metrics=[metric])

        try:
            return list(result.scores[0].values())[0]
        except Exception:
            return None
