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
    __slots__ = ("_data", "_iter")

    def __init__(self) -> None:
        self._data = cast(Optional[Dict[str, List[Any]]], None)

    def __repr__(self) -> str:
        return f"fastrepl.Dataset({{\n    features: {self.column_names},\n    num_rows: {self.__len__()}\n}})"

    def __len__(self) -> int:
        if self._data is None:
            return 0
        first_value = next(iter(self._data.values()), [])
        return len(first_value)

    def __iter__(self):
        self._iter = range(len(self._data) + 1).__iter__()
        return self

    def __next__(self):
        try:
            i = next(self._iter)
            return [v[i] for v in self._data.values()]
        except StopIteration:
            raise StopIteration

    @property
    def column_names(self) -> List[str]:
        if self._data is None:
            raise EmptyDatasetError
        return list(self._data.keys())

    def add_column(self, name: str, values: List[Any]) -> "Dataset":
        if self.__len__() != len(values):
            raise ValueError

        if self._data is None:
            raise EmptyDatasetError
        self._data[name] = values
        return self

    @classmethod
    def _headers(cls):
        return {"Authorization": f"Bearer {fastrepl.api_key}"}

    @classmethod
    def _base_url(cls):
        return f"{fastrepl.api_base}/dataset"

    @classmethod
    def from_dict(cls, data: Dict[str, List[Any]]) -> "Dataset":
        size = len(next(iter(data.values()), []))
        for value in data.values():
            if len(value) != size:
                raise ValueError

        ds = Dataset()
        ds._data = data
        return ds

    def to_dict(self) -> Dict[str, List[Any]]:
        if self._data is None:
            raise EmptyDatasetError

        return self._data

    @classmethod
    def from_hf(cls, data: Any) -> "Dataset":
        optional_package_import.check()
        hf_ds = cast(HF_Dataset, data)

        ds = Dataset()
        ds._data = hf_ds.to_dict()
        return ds

    def to_hf(self) -> HF_Dataset:
        optional_package_import.check()
        return HF_Dataset.from_dict(self._data)

    @classmethod
    def from_cloud(cls, id: str, version: Optional[str] = None) -> "Dataset":
        if fastrepl.api_key is None or fastrepl.api_base is None:
            raise ValueError

        url = f"{Dataset._base_url()}/get/{id}"
        if version is not None:
            url += f"?version={version}"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.get(url).json()
            data = res["data"]

            return Dataset.from_dict(data)

    @classmethod
    def list_cloud(cls) -> List[str]:
        if fastrepl.api_key is None or fastrepl.api_base is None:
            raise ValueError

        url = f"{Dataset._base_url()}/list"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.get(url)
            return res.json()["ids"]

    def push_to_cloud(self, id: Optional[str]) -> str:
        if fastrepl.api_key is None or fastrepl.api_base is None:
            raise ValueError

        url = f"{Dataset._base_url()}/new"

        with httpx.Client(headers=Dataset._headers()) as client:
            res = client.post(url, json={"id": id, "data": self._data})

            try:
                return res.json()["id"]
            except Exception as e:
                raise DatasetPushError(res)
