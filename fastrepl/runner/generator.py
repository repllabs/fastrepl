from typing import Optional

import time
import httpx
from fastrepl.dataset import Dataset

import fastrepl
from fastrepl.utils import getenv, console
from fastrepl.runner.base import BaseRunner

NUM_THREADS = getenv("NUM_THREADS", 8)


class LocalGeneratorRunner(BaseRunner):
    def __init__(self, generator: fastrepl.Generator) -> None:
        url = "https://github.com/repllabs/fastrepl/blob/521aec43fb01aaf8fadd6b9b20ef0823239ecece/exp/pg_essay_questions.ipynb"
        raise NotImplementedError(f"See {url!r} for reference.")

    def run(self) -> Dataset:
        raise NotImplementedError


class RemoteGeneratorRunner(BaseRunner):
    def __init__(self, generator: fastrepl.Generator) -> None:
        self._generator = generator

    def run(self, job_id: Optional[str] = None) -> Dataset:
        assert fastrepl.api_base is not None
        assert fastrepl.api_key is not None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {fastrepl.api_key}",
        }

        if job_id is None:
            new_res = httpx.post(
                f"{fastrepl.api_base}/question/generate/new",
                headers=headers,
                json={"text": self._generator.source},
            )

            if new_res.status_code != 200:
                raise Exception(new_res.json())

            job_id = new_res.json()["id"]

        client = httpx.Client(headers=headers)
        try:
            with console.status(f"[cyan3]Waiting for job '{job_id}' to finish..."):
                while True:
                    res = client.get(
                        f"{fastrepl.api_base}/question/generate/result/{job_id}",
                    )

                    if res.status_code not in [200, 202]:
                        raise Exception(res.json())

                    if res.status_code == 202:
                        time.sleep(0.5)
                        continue

                    return Dataset.from_dict({"question": res.json()})
        except KeyboardInterrupt:
            console.print(
                f"[cyan3]Interrupted!\nYou can retrieve the job status with `run(job_id='{job_id}')`"
            )
        finally:
            client.close()

        return Dataset.from_dict({})
