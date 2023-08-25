from typing import Optional

import pytest

run_url = Optional[str]


@pytest.hookimpl(tryfirst=True)
def pytest_addoption(parser):
    parser.addoption(
        "--fastrepl",
        action="store_true",
        default=False,
        help="Enable experimental fastrepl testing",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--fastrepl"):
        print("Initializing fastrepl session")


def pytest_sessionfinish(session):
    global run_url
    if session.config.getoption("--fastrepl"):
        run_url = "https://docs.fastrepl.com"


# TODO: We need to do some clever stuffs here
# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--fastrepl"):
#         for item in items:
#             print(item)
#             item.obj = fastrepl.test(item.obj)


def pytest_terminal_summary(terminalreporter):
    global run_url
    if run_url:
        terminalreporter.write_sep("=", "fastrepl summary", purple=True)
        terminalreporter.write_line(f"{run_url}")
