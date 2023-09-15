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
        model: SUPPORTED_MODELS,
        metric: RAGAS_METRICS,
    ) -> MetricWithLLM:
        llm = ChatLiteLLM(model=str(model))  # type: ignore[call-arg]

        if metric == "AnswerRelevancy":
            return AnswerRelevancy(llm=llm, batch_size=1)
        elif metric == "ContextRecall":
            return ContextRecall(llm=llm, batch_size=1)
        elif metric == "ContextRelevancy":
            return ContextRelevancy(llm=llm, batch_size=1)
        elif metric == "Faithfulness":
            return Faithfulness(llm=llm, batch_size=1)
        elif metric == "harmfulness":
            harmfulness.llm = llm
            return harmfulness
        elif metric == "maliciousness":
            maliciousness.llm = llm
            return maliciousness
        elif metric == "coherence":
            coherence.llm = llm
            return coherence
        elif metric == "correctness":
            correctness.llm = llm
            return correctness
        elif metric == "conciseness":
            conciseness.llm = llm
            return conciseness
        else:
            raise ValueError

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
