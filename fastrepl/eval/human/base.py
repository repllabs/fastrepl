from abc import ABC, abstractmethod


class BaseHumanEval(ABC):
    @abstractmethod
    def compute(self, sample: str, context: str) -> str:
        ...
