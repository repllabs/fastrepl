from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod


class BaseMetaEvalNode(ABC):
    @abstractmethod
    def compute(
        self, predictions: List[Any], references: List[Any], **kwargs
    ) -> Dict[str, Any]:
        ...


class BaseEvalNode(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs) -> Optional[Any]:
        ...


class BaseSimpleEvalNode(BaseEvalNode):
    @abstractmethod
    def compute(self, *, sample: str) -> Optional[str]:
        ...


class RAGEvalNode(BaseEvalNode):
    @abstractmethod
    def compute(
        self,
        *,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truths: List[str],
    ) -> Optional[float]:
        ...
