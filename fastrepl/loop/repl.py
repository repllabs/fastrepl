# https://rich.readthedocs.io/en/stable/live.html
# https://rich.readthedocs.io/en/stable/tables.html

from typing import List
from contextlib import ContextDecorator
import importlib.metadata

from rich.progress import Progress
from multiprocessing.pool import ThreadPool

from fastrepl.eval import Evaluator
from fastrepl.utils import getenv, get_cuid
from fastrepl.context import REPLContext


NUM_THREADS = getenv("NUM_THREADS", 8)
DEFAULT_INFO = {"fastrepl": importlib.metadata.version("fastrepl")}


class REPLController:
    __slots__ = ("id", "info", "_evaluator", "_display")

    def __init__(self):
        self.id = get_cuid()
        self.info = DEFAULT_INFO

    def set_evaluator(self, evaluator: Evaluator):
        self._evaluator = evaluator

    def eval(self, inputs: List[str]) -> List[str]:
        ret = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(inputs))

            with ThreadPool(NUM_THREADS) as pool:
                for result in pool.imap(self._evaluator.run, inputs):
                    ret.append(result)
                    progress.update(task, advance=1, refresh=True)
        return ret


class REPL(ContextDecorator):
    __slots__ = "controller"

    def __init__(self):
        self.controller = REPLController()

    def __enter__(self) -> REPLController:
        return self.controller

    def __exit__(self, *args):
        REPLContext.reset()
        self.controller = None
