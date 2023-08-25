from typing import Optional, List

import pytest
import _pytest.terminal


run_url: Optional[str] = None


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--fastrepl",
        action="store_true",
        default=False,
        help="Enable experimental fastrepl testing",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "fastrepl: this marker takes no arguments.")
    # config.addinivalue_line(
    #     "markers", "fastrepl(arg, arg2): this marker takes arguments."
    # )


def pytest_sessionstart(session: pytest.Session):
    if session.config.getoption("--fastrepl"):
        print("Initializing fastrepl session")


def pytest_sessionfinish(session: pytest.Session):
    global run_url
    if session.config.getoption("--fastrepl"):
        run_url = "https://docs.fastrepl.com"


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]):
    for item in items:
        for marker in item.iter_markers():
            if marker.name != "fastrepl":
                continue

            if config.getoption("--fastrepl"):
                # TODO: We need to do some clever stuffs here
                # item.obj = fastrepl.test(item.obj)
                pass
            else:
                item.add_marker(
                    pytest.mark.skip(
                        "test marked with fastrepl will be skipped without --fastrepl"
                    )
                )


def pytest_terminal_summary(terminalreporter: _pytest.terminal.TerminalReporter):
    global run_url
    if run_url:
        terminalreporter.write_sep("=", "fastrepl summary", purple=True)
        terminalreporter.write_line(f"{run_url}")
