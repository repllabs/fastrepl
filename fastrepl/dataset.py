from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Any, cast

import httpx
from lazy_imports import try_import

with try_import() as optional_hf:
    from datasets import Dataset as HF_Dataset

with try_import() as optional_lf:
    from langfuse.client import DatasetClient as LF_Dataset

if TYPE_CHECKING:
    from datasets import Dataset as HF_Dataset
    from langfuse.client import DatasetClient as LF_Dataset

import fastrepl
from fastrepl.errors import DatasetPushError


class Dataset:
    __slots__ = ("_data", "_iter")

    def __init__(self) -> None:
        self._data: Dict[str, List[Any]] = {}

    def __repr__(self) -> str:
        return f"fastrepl.Dataset({{\n    features: {self.column_names},\n    num_rows: {self.__len__()}\n}})"

    def __len__(self) -> int:
        v0: List[Any] = next(iter(self._data.values()), [])
        return len(v0)

    def __getitem__(self, key: str) -> List[Any]:
        if key in self._data:
            return self._data[key]
        else:
            raise KeyError

    def __iter__(self):
        key = list(self._data.keys())[0]
        self._iter = range(len(self._data[key])).__iter__()
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            i = next(self._iter)
            return {col: self._data[col][i] for col in self.column_names}
        except StopIteration:
            raise StopIteration

    @property
    def column_names(self) -> List[str]:
        return list(self._data.keys())

    def add_column(self, name: str, values: List[Any]) -> "Dataset":
        if self.__len__() != len(values):
            raise ValueError

        self._data[name] = list(values)
        return self

    def rename_column(self, old_name: str, new_name: str) -> "Dataset":
        data = {col: self._data[col] for col in self.column_names}
        data[new_name] = data.pop(old_name)
        return Dataset.from_dict(data)

    def remove_column(self, name: str) -> "Dataset":
        data = {col: self._data[col] for col in self.column_names}
        del data[name]
        return Dataset.from_dict(data)

    def map(self, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> "Dataset":
        rows = []
        for row in self:
            rows.append(func(row))
        data = {col: [row[col] for row in rows] for col in self.column_names}
        return Dataset.from_dict(data)

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
        return self._data

    @classmethod
    def from_hf(cls, data: Any) -> "Dataset":
        optional_hf.check()
        hf_ds = cast(HF_Dataset, data)

        ds = Dataset()
        ds._data = hf_ds.to_dict()
        return ds

    def to_hf(self) -> HF_Dataset:
        optional_hf.check()
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

    @classmethod
    def from_langfuse(cls, data: Any) -> "Dataset":
        optional_lf.check()
        lf_ds = cast(LF_Dataset, data)

        PREFIX = "_lf"

        data = {
            f"{PREFIX}_item_index": [],
            "inputs": [],
            "expected_outputs": [],
        }

        for i, item in enumerate(lf_ds.items):
            data[f"{PREFIX}_item_index"].append(i)
            data["inputs"].append(item.input)
            data["expected_outputs"].append(item.expected_output)

        return Dataset.from_dict(data)
