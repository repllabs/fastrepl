from typing import List, Any

import json
import pytest

from fastrepl.utils import getenv


class TestReport:
    @staticmethod
    def add(data: Any) -> None:
        s = json.dumps(data)
        # NOTE: This will be later parsed by fastrepl
        print(f"__FASTREPL_START_{s}_FASTREPL_END__")


@pytest.fixture(scope="session")
def report():
    return TestReport


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--fastrepl",
        action="store_true",
        default=False,
        help="Enable experimental fastrepl evaluation runner",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fastrepl: experiemental fastrepl testing")


def pytest_sessionstart(session: pytest.Session):
    if session.config.getoption("--fastrepl"):
        import litellm

        # NOTE: This will be provided in Github App
        api_base = getenv("LITELLM_PROXY_API_BASE", "")
        if api_base != "":  # pragma: no cover
            api_key = getenv("LITELLM_PROXY_API_KEY", "")

            print("LITELLM_PROXY_API_BASE is set")
            if api_key != "":
                print("LITELLM_PROXY_API_KEY is set")

            litellm.api_base = api_base
            litellm.headers = {"Authorization": api_key}
        else:
            print("LITELLM_PROXY_API_BASE is not set")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]):
    if not config.getoption("--fastrepl"):
        for item in items:
            for marker in item.iter_markers():
                if marker.name == "fastrepl":
                    item.add_marker(
                        pytest.mark.skip(
                            "--fastrepl is not specified, skipping tests with fastrepl marker"
                        )
                    )
