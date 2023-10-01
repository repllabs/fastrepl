from abc import ABC, abstractmethod
from typing import Optional, List

from fastrepl.eval.base import BaseSimpleEvalNode, BaseRAGEvalNode


class Evaluator(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        ...

    @abstractmethod
    def inputs(self) -> List[str]:
        ...


class SimpleEvaluator(Evaluator):
    def __init__(self, node: BaseSimpleEvalNode) -> None:
        self.node = node

    def run(self, *, sample: str) -> Optional[str]:
        return self.node.run(sample=sample)

    def inputs(self) -> List[str]:
        return self.node.inputs()


class RAGEvaluator(Evaluator):
    def __init__(self, node: BaseRAGEvalNode) -> None:
        self.node = node

    def run(
        self,
        *,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Optional[float]:
        return self.node.run(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truths=ground_truths,
        )

    def inputs(self) -> List[str]:
        return self.node.inputs()
