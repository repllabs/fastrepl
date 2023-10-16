from typing import Optional

import time
import httpx
from fastrepl.dataset import Dataset

import fastrepl
from fastrepl.utils import console
from fastrepl.runner.base import BaseRunner


class LocalGeneratorRunner(BaseRunner):
    def __init__(self, generator: fastrepl.BaseGenerator) -> None:
        self._generator = generator

    def run(self) -> Dataset:
        if self._generator.kind == "question":
            return self._run_question_generator()
        else:
            raise NotImplementedError

    def _run_question_generator(self) -> Dataset:
        try:
            from llama_index.evaluation import DatasetGenerator
            from llama_index.schema import Document
            import spacy  # type: ignore
        except ImportError:
            console.print(
                """
[cyan3]You need `llama_index` and `spacy` installed to use generate questions locally.
Note that it is recommended to use `remote_runner` for generating questions.
Read more about it: `https://docs.fastrepl.com/get_started/dataset_generation#remote`
                """
            )
            return Dataset.from_dict({"question": []})

        data_generator = DatasetGenerator.from_documents(
            [Document(text=self._generator.source)],
            num_questions_per_chunk=5,
        )

        import nest_asyncio

        nest_asyncio.apply()

        questions = data_generator.generate_questions_from_nodes()
        return Dataset.from_dict({"question": questions})


class RemoteGeneratorRunner(BaseRunner):
    def __init__(self, generator: fastrepl.BaseGenerator) -> None:
        self._generator = generator

    def run(self, job_id: Optional[str] = None) -> Dataset:
        assert fastrepl.api_base is not None
        assert fastrepl.api_key is not None

        if self._generator.kind == "question":
            return self._run_question_generator(job_id)
        else:
            raise NotImplementedError

    def _run_question_generator(self, job_id: Optional[str] = None) -> Dataset:
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
