import inspect

from fastrepl import Updatable as _Updatable

from fastrepl.utils import LocalContext
from fastrepl.repl.context import REPLContext


class Updatable(_Updatable):
    def __init__(
        self,
        key: str,
        value: str,
        what="this is updatable value",
        how="be creative while maintaining the original meaning",
    ):
        super().__init__(key, value)

        self.what, self.how = what, how
        self._ctx = LocalContext(inspect.stack()[1])

        REPLContext.trace(self._ctx, self._key, value)

    @property
    def value(self) -> str:
        return REPLContext.get_current(self._ctx, self._key)
