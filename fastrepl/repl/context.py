from typing import (
    ClassVar,
    Tuple,
    List,
    Dict,
    OrderedDict,
    DefaultDict,
)

import rich
import graphviz

from fastrepl.utils import LocalContext, get_cuid, GraphInfo, build_graph


def set_status(status: str):
    REPLContext.set_status(status)


def update(pairs: List[Tuple[str, str]]):
    REPLContext.update(pairs)


def visualize() -> graphviz.Digraph:
    nodes: List[Tuple[str, str]] = []
    edges: List[Tuple[str, str]] = []

    return build_graph(GraphInfo(id=REPLContext._status, nodes=nodes, edges=edges))


class REPLContext:
    """single context that tracks all experiments in a REPL"""

    _status: ClassVar[str] = get_cuid()
    """current status of the REPL"""
    _history: ClassVar[List[str]] = [_status]
    """history of REPL status"""
    # fmt: off
    _trace: Dict[LocalContext, Dict[str, OrderedDict[str, str]]] = DefaultDict(lambda: DefaultDict(OrderedDict))
    """mapping: ctx -> key -> status -> value"""
    # fmt: on

    @staticmethod
    def reset():
        REPLContext._status = get_cuid()
        REPLContext._history = [REPLContext._status]
        REPLContext._trace = DefaultDict(lambda: DefaultDict(OrderedDict))

    @staticmethod
    def trace(ctx: LocalContext, key: str, value: str):
        for key_status_value in REPLContext._trace.values():
            if key in key_status_value.keys():
                raise ValueError(f"{key!r} already exists")

        REPLContext._trace[ctx][key][REPLContext._status] = value

    @staticmethod
    def update(pairs: List[Tuple[str, str]]):
        next_status = get_cuid()

        for key_status_value in REPLContext._trace.values():
            for key, status_value in key_status_value.items():
                try:
                    _, new_value = next((k, v) for k, v in pairs if key == k)
                    status_value[next_status] = new_value
                except StopIteration:
                    status_value[next_status] = status_value[REPLContext._status]

        REPLContext._history.append(next_status)
        REPLContext._status = next_status

    @staticmethod
    def set_status(status: str):
        if status not in REPLContext._history:
            raise ValueError(f"{status!r} is not valid status")

        REPLContext._status = status

    @staticmethod
    def get_current(ctx: LocalContext, key: str) -> str:
        return REPLContext._trace[ctx][key][REPLContext._status]
