import pytest
from _pytest.fixtures import FixtureRequest

import json
from sqlalchemy import create_engine

import fastrepl
from fastrepl.run.cache import SQLAlchemyCache


def get_sqlite_cache() -> SQLAlchemyCache:
    return SQLAlchemyCache(engine=create_engine("sqlite://"))


CACHE_OPTIONS = [
    get_sqlite_cache,
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest):
    cache_instance = request.param
    fastrepl.cache = cache_instance()
    if fastrepl.cache:
        fastrepl.cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")

    yield

    if fastrepl.cache:
        fastrepl.cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")


class TestCache:
    def test_basic(self):
        if fastrepl.cache:
            fastrepl.cache.update("model", "prompt", "response")
            assert fastrepl.cache.lookup("model", "prompt") == "response"
            assert fastrepl.cache.lookup("model", "prompt2") is None

            fastrepl.cache.clear()
            assert fastrepl.cache.lookup("model", "prompt") is None
        else:
            raise ValueError("Cache not set. This should never happen.")

    def test_json(self):
        if fastrepl.cache:
            msgs = [
                {
                    "role": "user",
                    "text": "Hello, how are you?",
                }
            ]
            res = {"text": "1"}
            fastrepl.cache.update("model", json.dumps(msgs), json.dumps(res))
            assert fastrepl.cache.lookup("model", json.dumps(msgs)) == json.dumps(res)
        else:
            raise ValueError("Cache not set. This should never happen.")
