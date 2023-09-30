from abc import ABC, abstractmethod

from fastrepl.dataset import Dataset


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> Dataset:
        pass
