from abc import ABC, abstractmethod

from datasets import Dataset


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> Dataset:
        pass
