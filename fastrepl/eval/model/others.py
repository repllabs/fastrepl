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

        match metric:
            case "AnswerRelevancy":
                return AnswerRelevancy(llm=llm, batch_size=1)
            case "ContextRecall":
                return ContextRecall(llm=llm, batch_size=1)
            case "ContextRelevancy":
                return ContextRelevancy(llm=llm, batch_size=1)
            case "Faithfulness":
                return Faithfulness(llm=llm, batch_size=1)
            case "harmfulness":
                harmfulness.llm = llm
                return harmfulness
            case "maliciousness":
                maliciousness.llm = llm
                return maliciousness
            case "coherence":
                coherence.llm = llm
                return coherence
            case "correctness":
                correctness.llm = llm
                return correctness
            case "conciseness":
                conciseness.llm = llm
                return conciseness

    def compute(self, prediction: str, context: Any) -> Any:
        ds: Dataset

        match self.metric.evaluation_mode:
            case EvaluationMode.qac:
                ds = Dataset.from_dict(
                    {
                        "question": "",
                        "answer": prediction,
                        "contexts": [],
                    }
                )
            case EvaluationMode.qa:
                ds = Dataset.from_dict(
                    {
                        "question": "",
                        "answer": prediction,
                    }
                )
            case EvaluationMode.qc:
                ds = Dataset.from_dict(
                    {
                        "question": "",
                        "contexts": [],
                    }
                )
            case EvaluationMode.gc:
                ds = Dataset.from_dict(
                    {
                        "ground_truths": [],
                        "contexts": [],
                    }
                )

        result = evaluate(ds, metrics=[self.metric])
        return result
