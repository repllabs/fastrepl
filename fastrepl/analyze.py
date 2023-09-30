from typing import Literal, Dict

from fastrepl.dataset import Dataset
from fastrepl.utils import kappa


class Analyzer:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    # TODO: only Kappa for now
    def run(self, mode: Literal["kappa"], feature="result") -> Dict[str, float]:
        assert mode == "kappa"

        if not isinstance(self._dataset[feature], list):
            return {}

        return {"kappa": kappa(self._dataset[feature])}
