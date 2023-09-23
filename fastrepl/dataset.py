from typing import Literal, Optional, List, overload

import time
import requests
from datasets import Dataset as HuggingfaceDataset

import fastrepl
from fastrepl.utils import console


class Dataset:
    @overload
    @classmethod
    def generate_from(
        cls, kind: Literal["question"], *, job_id: str
    ) -> HuggingfaceDataset:
        ...

    @overload
    @classmethod
    def generate_from(
        cls, kind: Literal["question"], *, source: List[str]
    ) -> HuggingfaceDataset:
        ...

    @classmethod
    def generate_from(
        cls,
        kind: Literal["question"],
        *,
        source: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> HuggingfaceDataset:
        assert kind == "question"  # TODO

        if job_id is None:
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

            job_id = new_res.json()["id"]

        with console.status(f"[cyan3]Waiting for job '{job_id}' to finish..."):
            while True:
                res = requests.get(
                    f"{fastrepl.api_base}/question/generate/result/{job_id}",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {fastrepl.api_key}",
                    },
                )

                if res.status_code not in [200, 202]:
                    raise Exception(res.json())

                if res.status_code == 202:
                    time.sleep(0.5)
                    continue

                return HuggingfaceDataset.from_dict(
                    {"question": res.json()},
                )
