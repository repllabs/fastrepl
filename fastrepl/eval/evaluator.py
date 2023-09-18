from abc import ABC, abstractmethod
from typing import Optional, List

from fastrepl.eval.base import BaseSimpleEvalNode, RAGEvalNode


class Evaluator(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class SimpleEvaluator(Evaluator):
    def __init__(self, node: BaseSimpleEvalNode) -> None:
        self.node = node

    def run(self, *, sample: str) -> Optional[str]:
        return self.node.compute(sample=sample)


class RAGEvaluator(Evaluator):
    def __init__(self, node: RAGEvalNode) -> None:
        self.node = node

    def run(
        self,
        *,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truths: List[str],
    ) -> Optional[float]:
        return self.node.compute(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truths=ground_truths,
        )
