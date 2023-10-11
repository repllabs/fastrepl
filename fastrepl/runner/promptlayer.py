import os
from typing import Optional

from fastrepl.dataset import Dataset
from fastrepl.eval import Evaluator
from fastrepl.runner.base import BaseRunner
from fastrepl.utils import map_number_range

import httpx


class PromptLayerRunner(BaseRunner):
    def __init__(self, evaluator: Evaluator, api_key: Optional[str] = None) -> None:
        self._evaluator = evaluator
        self.api_key = os.environ.get("PL_API_KEY") if api_key is None else api_key

    def run(self, ds: Dataset):
        with httpx.Client() as client:
            for row in ds:
                try:
                    request_id = row.pop("request_id")
                except KeyError as e:
                    raise KeyError("'request_id' is required but not found") from e

                # TODO
                result = float(self._evaluator.run(**row))
                # result = map_number_range(...)
                result = round(result)

                client.post(
                    "https://api.promptlayer.com/rest/track-score",
                    json={
                        "request_id": request_id,
                        "score": result,
                        "api_key": self.api_key,
                    },
                )
