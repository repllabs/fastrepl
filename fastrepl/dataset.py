from typing import TYPE_CHECKING, Optional, Dict, List, Any, cast

import httpx
from lazy_imports import try_import

with try_import() as optional_package_import:
    from datasets import Dataset as HF_Dataset

if TYPE_CHECKING:
    from datasets import Dataset as HF_Dataset


import fastrepl
from fastrepl.errors import EmptyDatasetError, DatasetPushError


class Dataset:
    __slots__ = ("data",)

    def __init__(self) -> None:
        self.data = cast(Optional[Dict[str, List[Any]]], None)

        if fastrepl.api_key is None or fastrepl.api_base is None:
            raise ValueError

    def __repr__(self) -> str:
        return ""

    @classmethod
    def _headers(cls):
        return {"Authorization": f"Bearer {fastrepl.api_key}"}

    @classmethod
    def _base_url(cls):
        return f"{fastrepl.api_base}/dataset"

    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]) -> "Dataset":
        ds = Dataset()
        ds.data = data
        return ds

    def to_dict(self) -> Dict[str, List[Any]]:
        if self.data is None:
            raise EmptyDatasetError

        return self.data

    @classmethod
    def from_hf(cls, data: Any) -> "Dataset":
        optional_package_import.check()
        hf_ds = cast(HF_Dataset, data)

        ds = Dataset()
        ds.data = hf_ds.to_dict()
        return ds

    def to_hf(self) -> HF_Dataset:
        optional_package_import.check()
        return HF_Dataset.from_dict(self.data)

    @classmethod
    def from_cloud(cls, id: str, version: Optional[str] = None) -> "Dataset":
        url = f"{Dataset._base_url()}/get/{id}"
        if version is not None:
            url += f"?version={version}"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.get(url).json()
            data = res["data"]

            return Dataset.from_dict(data)

    @classmethod
    def list_cloud(cls) -> List[str]:
        url = f"{Dataset._base_url()}/list"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.get(url)
            return res.json()["ids"]

    def push_to_cloud(self, id: Optional[str]) -> str:
        url = f"{Dataset._base_url()}/new"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.post(url, json={"id": id, "data": self.data})

            try:
                return res.json()["id"]
            except Exception as e:
                raise DatasetPushError(res)
