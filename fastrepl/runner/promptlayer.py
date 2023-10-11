from typing import Optional

import os
import threading

from fastrepl.dataset import Dataset
from fastrepl.eval import Evaluator
from fastrepl.runner.base import BaseRunner
from fastrepl.utils import map_number_range

import httpx


class PromptLayerRunner(BaseRunner):
    def __init__(self, evaluator: Evaluator, api_key: Optional[str] = None) -> None:
        self._evaluator = evaluator
        self.api_key = os.environ.get("PL_API_KEY") if api_key is None else api_key

    def _run(self, ds: Dataset):
        with httpx.Client() as client:
            for row in ds:
                try:
                    request_id = row.pop("request_id")
                except KeyError as e:
                    raise KeyError("'request_id' is required but not found") from e

                result = float(self._evaluator.run(**row))

                try:
                    from_min, from_max = (  # TODO: Better typing
                        self._evaluator.node.to_min,  # type: ignore[attr-defined]
                        self._evaluator.node.to_max,  # type: ignore[attr-defined]
                    )
                except AttributeError:  # RAGAS
                    from_min, from_max = 0, 1
                finally:
                    result = map_number_range(result, from_min, from_max, 0, 100)

                # https://docs.promptlayer.com/reference/track-score
                client.post(
                    "https://api.promptlayer.com/rest/track-score",
                    json={
                        "request_id": request_id,
                        "score": round(result),
                        "api_key": self.api_key,
                    },
                )

    def run(self, ds: Dataset, use_threading=True):
        if use_threading:
            thread = threading.Thread(target=self._run, args=(ds,))
            thread.start()
        else:
            self._run(ds)
