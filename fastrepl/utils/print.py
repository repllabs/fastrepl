import os, contextlib
from rich.console import Console

console = Console()


def suppress(fn):  # pragma: no cover
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    return fn(*args, **kwargs)

    return wrapper
