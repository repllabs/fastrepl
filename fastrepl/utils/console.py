from rich.console import Console

console = Console()

import os, contextlib


def no_stdout(fn):  # pragma: no cover
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return fn(*args, **kwargs)

    return wrapper


def no_stderr(fn):  # pragma: no cover
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                return fn(*args, **kwargs)

    return wrapper
