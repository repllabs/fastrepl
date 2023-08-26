from fastrepl.polish import Updatable as Updatable_normal
from fastrepl.repl.polish import Updatable as Updatable_repl


def fn_without_updatable():
    return "long text" * 100


def fn_with_updatable_normal():
    return Updatable_normal(key="test", value="long text" * 100)


def fn_with_updatable_repl():
    return Updatable_repl(key="test", value="long text" * 100)


def test_fn_without_updatable(benchmark):
    benchmark(fn_without_updatable)


def test_fn_with_updatable_normal(benchmark):
    benchmark(fn_with_updatable_normal)


def test_fn_with_updatable_repl(benchmark):
    benchmark(fn_with_updatable_repl)
