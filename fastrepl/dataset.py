from typing import Literal, List

import time
import requests
from datasets import Dataset as HuggingfaceDataset

import fastrepl
from fastrepl.utils import console


class Dataset:
    @classmethod
    def generate_from(
        cls, kind: Literal["question"], source: List[str]
    ) -> HuggingfaceDataset:
        assert kind == "question"

        new_res = requests.post(
            f"{fastrepl.api_base}/question/generate/new",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {fastrepl.api_key}",
            },
            json={"texts": source},
        )

        if new_res.status_code != 200:
            raise Exception(new_res.json())

        id = new_res.json()["id"]

        def get() -> requests.Response:
            return requests.get(
                f"{fastrepl.api_base}/question/generate/result/{id}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {fastrepl.api_key}",
                },
            )

        with console.status("Waiting for result..."):
            while True:
                res = get()
                if res.status_code == 202:
                    time.sleep(0.5)
                    continue

                if res.status_code == 200:
                    return res.json()

                raise Exception(res.json())
