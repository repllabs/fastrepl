from typing import Literal, List

import requests
from datasets import Dataset as HuggingfaceDataset

import fastrepl


class Dataset:
    @classmethod
    def generate_from(
        cls, kind: Literal["question"], source: List[str]
    ) -> HuggingfaceDataset:
        assert kind == "question"

        res = requests.post(
            f"{fastrepl.api_base}/question/generate/new",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {fastrepl.api_key}",
            },
            json={"texts": ["hello", "world"]},
        )

        id = res.json()["id"]

        return id
