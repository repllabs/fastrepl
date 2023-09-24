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
        cls, kind: Literal["question"], *, source: str
    ) -> HuggingfaceDataset:
        ...

    @classmethod
    def generate_from(
        cls,
        kind: Literal["question"],
        *,
        source: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> HuggingfaceDataset:
        if fastrepl.api_key is None:
            return cls._local_generate_from(kind, source=source, job_id=job_id)
        return cls._cloud_generate_from(kind, source=source, job_id=job_id)

    @classmethod
    def _cloud_generate_from(
        cls,
        kind: Literal["question"],
        *,
        source: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> HuggingfaceDataset:
        if job_id is None:
            new_res = requests.post(
                f"{fastrepl.api_base}/question/generate/new",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {fastrepl.api_key}",
                },
                json={"text": source},
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

    @classmethod
    def _local_generate_from(
        cls,
        kind: Literal["question"],
        *,
        source: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> HuggingfaceDataset:
        url = "https://github.com/repllabs/fastrepl/blob/521aec43fb01aaf8fadd6b9b20ef0823239ecece/exp/pg_essay_questions.ipynb"
        raise NotImplementedError(f"See {url} for reference.")
