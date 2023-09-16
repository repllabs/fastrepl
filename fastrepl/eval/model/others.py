from typing import Literal, Any

from datasets import Dataset
from langchain.chat_models import ChatLiteLLM

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

from fastrepl.eval.base import BaseEvalNode
from fastrepl.llm import SUPPORTED_MODELS


class RagasEvaluation(BaseEvalNode):
    def __init__(
        self,
        metric: RAGAS_METRICS,
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
    ):
        optional_package_import.check()

        self.metric = self._load_metric(model, metric)

    def _load_metric(
        self,
        model_name: SUPPORTED_MODELS,
        metric_name: RAGAS_METRICS,
    ) -> MetricWithLLM:
        # TODO
        llm = ChatLiteLLM(model=str(model_name))  # type: ignore[call-arg]

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

    def compute(self, prediction: str, context: Any) -> Any:
        ds: Dataset

        if self.metric.evaluation_mode == EvaluationMode.qac:
            ds = Dataset.from_dict(
                {
                    "question": "",
                    "answer": prediction,
                    "contexts": [],
                }
            )
        elif self.metric.evaluation_mode == EvaluationMode.qa:
            ds = Dataset.from_dict(
                {
                    "question": "",
                    "answer": prediction,
                }
            )
        elif self.metric.evaluation_mode == EvaluationMode.qc:
            ds = Dataset.from_dict(
                {
                    "question": "",
                    "contexts": [],
                }
            )
        elif self.metric.evaluation_mode == EvaluationMode.gc:
            ds = Dataset.from_dict(
                {
                    "ground_truths": [],
                    "contexts": [],
                }
            )
        else:
            raise ValueError

        result = evaluate(dataset=ds, metrics=[self.metric])
        return result
