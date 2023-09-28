import pytest
from pytest_httpx import HTTPXMock
from datasets import Dataset as HF_Dataset

import fastrepl

fastrepl.api_key = "TEST"
fastrepl.api_base = "http://api.fastrepl.com"


def test_no_api_key():
    fastrepl.api_key = None

    with pytest.raises(Exception):
        fastrepl.Dataset.from_dict({"a": [1]})


def test_from_dict():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_to_dict():
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_from_hf():
    hf_ds = HF_Dataset.from_dict({"a": [1]})
    ds = fastrepl.Dataset.from_hf(hf_ds)
    assert ds.to_dict() == {"a": [1]}


def test_from_cloud(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"a": ["1"]})

    ds = fastrepl.Dataset.from_cloud(id="123")
    assert ds.data == {"a": ["1"]}
