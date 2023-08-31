from typing import Any, List, Union
from abc import ABC, abstractmethod


class BaseEvalWithReference(ABC):
    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs):
        ...


class BaseEvalWithoutReference(ABC):
    @abstractmethod
    def compute(self, prediction: str, context: str) -> Union[None, int, str]:
        ...

    @abstractmethod
    def is_interactive(self) -> bool:
        ...
