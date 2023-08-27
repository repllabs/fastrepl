from fastrepl import Updatable
from fastrepl.repl import Updatable as UpdatableREPL


def fn_without_updatable():
    return "long text" * 100


def fn_with_updatable():
    return Updatable(key="test", value="long text" * 100)


def fn_with_updatable_repl():
    return UpdatableREPL(key="test", value="long text" * 100)


def test_fn_without_updatable(benchmark):
    benchmark(fn_without_updatable)


def test_fn_with_updatable(benchmark):
    benchmark(fn_with_updatable)


def test_fn_with_updatable_repl(benchmark):
    benchmark(fn_with_updatable_repl)
