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

    @abstractmethod
    def inputs(self) -> List[str]:
        ...


class BaseSimpleEvalNode(BaseEvalNode):
    @abstractmethod
    def compute(self, *, sample: str) -> Optional[str]:
        ...

    def inputs(self) -> List[str]:
        return ["sample"]


class RAGEvalNode(BaseEvalNode):
    @abstractmethod
    def compute(
        self,
        *,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
    ) -> Optional[float]:
        ...
