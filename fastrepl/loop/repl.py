# https://rich.readthedocs.io/en/stable/live.html
# https://rich.readthedocs.io/en/stable/tables.html

from typing import List
from contextlib import ContextDecorator
import time
import random
import importlib.metadata

from rich import box
from rich.live import Live
from rich.table import Table

from fastrepl.utils import get_cuid
from fastrepl.context import REPLContext

from fastrepl.eval import Evaluator


def generate_table() -> Table:
    table = Table(title="fastrepl", box=box.MINIMAL)
    table.add_column("ID")
    table.add_column("Value")
    table.add_column("Status")

    for row in range(random.randint(2, 6)):
        value = random.random() * 100
        table.add_row(
            f"{row}", f"{value:3.2f}", "[red]ERROR" if value < 50 else "[green]SUCCESS"
        )
    return table


DEFAULT_INFO = {"fastrepl": importlib.metadata.version("fastrepl")}


class REPLController:
    __slots__ = ("id", "info", "_evaluators", "_display")

    def __init__(self):
        self.id = get_cuid()
        self.info = DEFAULT_INFO

    def _set_display(self, live: Live):
        self._display = live

    def refresh(self):
        self._display.update(generate_table())

    def set_evaluators(self, evaluators: List[Evaluator]):
        self._evaluators = evaluators

    def run(self):
        for eval in self._evaluators:
            eval.run()  # TODO: We need to refresh it inside eval. should we pass it inside?
            # This might be due to design flaw. REPL controller is not actually controlling anything.


class REPL(ContextDecorator):
    __slots__ = "controller"

    def __init__(self):
        self.controller = REPLController()

    def __enter__(self) -> REPLController:
        live = Live(generate_table(), refresh_per_second=4).__enter__()
        self.controller._set_display(live)
        return self.controller

    def __exit__(self, *args):
        REPLContext.reset()
        self.controller._display.__exit__(*args)
        self.controller = None
