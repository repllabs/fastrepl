import pytest
import json

from fastrepl.run.cache import SQLiteCache


@pytest.fixture
def cache():
    cache = SQLiteCache()
    cache.clear()
    return cache


class TestSQLiteCache:
    def test_basic(self, cache: SQLiteCache):
        cache.update("model", "prompt", "response")
        assert cache.lookup("model", "prompt") == "response"
        assert cache.lookup("model", "prompt2") is None

        cache.clear()
        assert cache.lookup("model", "prompt") is None

    def test_json(self, cache: SQLiteCache):
        msgs = [
            {
                "role": "user",
                "text": "Hello, how are you?",
            }
        ]
        res = {"text": "1"}
        cache.update("model", json.dumps(msgs), json.dumps(res))
        assert cache.lookup("model", json.dumps(msgs)) == json.dumps(res)
