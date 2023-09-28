import pytest
from pytest_httpx import HTTPXMock
from datasets import Dataset as HF_Dataset

import fastrepl


@pytest.fixture
def mock_api():
    fastrepl.api_key = "TEST"
    fastrepl.api_base = "http://api.fastrepl.com"


def test_no_api_key(mock_api):
    fastrepl.api_key = None

    with pytest.raises(Exception):
        fastrepl.Dataset.from_dict({"a": [1]})


def test_from_dict(mock_api):
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_to_dict(mock_api):
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    assert ds.to_dict() == {"a": [1]}


def test_from_hf(mock_api):
    hf_ds = HF_Dataset.from_dict({"a": [1]})
    ds = fastrepl.Dataset.from_hf(hf_ds)
    assert ds.to_dict() == {"a": [1]}


def test_to_hf(mock_api):
    ds = fastrepl.Dataset.from_dict({"a": [1]})
    hf_ds = ds.to_hf()
    assert hf_ds.to_dict() == {"a": [1]}


def test_from_cloud(mock_api, httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"a": ["1"]})

    ds = fastrepl.Dataset.from_cloud(id="123")
    assert ds.data == {"a": ["1"]}


def test_push_to_cloud(mock_api, httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"id": "123"})

    ds = fastrepl.Dataset.from_dict({"a": [1]})
    id = ds.push_to_cloud(id="123")
    assert id == "123"
