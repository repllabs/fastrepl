import pytest
from pytest_httpx import HTTPXMock
from datasets import Dataset as HF_Dataset

import fastrepl


@pytest.fixture(autouse=True)
def mock_api():
    fastrepl.api_key = "TEST"
    fastrepl.api_base = "http://api.fastrepl.com"


def test_no_api_key():
    fastrepl.api_key = None

    with pytest.raises(Exception):
        fastrepl.Dataset.from_dict({"a": [1]})


def test_from_dict():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_from_dict_invalid():
    with pytest.raises(ValueError):
        fastrepl.Dataset.from_dict({"a": [1], "b": [2, 2]})


def test_to_dict():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_from_hf():
    hf_ds = HF_Dataset.from_dict({"a": [1]})
    ds = fastrepl.Dataset.from_hf(hf_ds)
    assert ds.to_dict() == {"a": [1]}


def test_to_hf():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    hf_ds = ds.to_hf()
    assert hf_ds.to_dict() == {"a": [1]}


def test_from_cloud(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"data": {"a": ["1"]}})

    ds = fastrepl.Dataset.from_cloud(id="123")
    assert ds._data == {"a": ["1"]}


def test_push_to_cloud(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"id": "123"})

    ds = fastrepl.Dataset.from_dict({"a": [1]})
    id = ds.push_to_cloud(id="123")
    assert id == "123"


def test_len():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert len(ds) == 1
    ds = fastrepl.Dataset.from_dict({"a": [0, 0, 0], "b": [1, 1, 1]})
    assert len(ds) == 3


def test_iter():
    ds = fastrepl.Dataset.from_dict({"a": [0, 0, 0], "b": [1, 1, 1]})

    count = 0
    for a, b in ds:
        assert a == 0
        assert b == 1
        count += 1
    assert count == 3


def test_column_names():
    ds = fastrepl.Dataset.from_dict({"a": [0, 0, 0], "b": [1, 1, 1]})
    assert ds.column_names == ["a", "b"]


def test_add_column():
    ds = fastrepl.Dataset.from_dict({"a": [0, 0, 0], "b": [1, 1, 1]})
    ds.add_column("c", [2, 2, 2])
    assert ds.column_names == ["a", "b", "c"]


def test_add_column_invalid():
    ds = fastrepl.Dataset.from_dict({"a": [0, 0, 0], "b": [1, 1, 1]})
    with pytest.raises(ValueError):
        ds.add_column("c", [2, 2, 2, 2])
