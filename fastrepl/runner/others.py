from typing import Callable, Optional, Iterable, Mapping, List, Any, cast

from concurrent.futures import ThreadPoolExecutor, Future
from rich.progress import Progress

import fastrepl
from fastrepl.utils import getenv, console

NUM_THREADS = getenv("NUM_THREADS", 8)


class LocalCustomRunner:
    def __init__(
        self,
        fn: Callable,
        output_feature="sample",
    ) -> None:
        self._fn = fn
        self._output_feature = output_feature

    def _run_single(
        self,
        args_list: List[Iterable[Any]],
        kwds_list: List[Mapping[str, Any]],
        cb: Callable[[Future], None],
    ) -> List[Any]:
        with ThreadPoolExecutor(min(NUM_THREADS, len(args_list))) as executor:
            futures: List[Future] = []

            for args, kwds in zip(args_list, kwds_list):
                future = executor.submit(self._fn, *args, **kwds)
                future.add_done_callback(cb)
                futures.append(future)

            return [future.result() for future in futures]

    def run(
        self,
        args_list: Optional[List[Iterable[Any]]] = None,
        kwds_list: Optional[List[Mapping[str, Any]]] = None,
        num=1,
        show_progress=True,
    ) -> fastrepl.Dataset:
        assert args_list is not None or kwds_list is not None

        args_list = args_list or [()] * len(cast(List[Iterable[Any]], kwds_list))
        kwds_list = kwds_list or [{}] * len(cast(List[Mapping[str, Any]], args_list))

        disable = not show_progress

        with Progress(console=console, disable=disable) as progress:
            msg = "[cyan]Processing..."
            task_id = progress.add_task(msg, total=len(args_list))
            cb = lambda future: progress.update(task_id, advance=1, refresh=True)

            if num > 1:
                results = [
                    self._run_single(args_list, kwds_list, cb) for _ in range(num)
                ]
                data = [list(item) for item in zip(*results)]
            else:
                data = self._run_single(args_list, kwds_list, cb)

            return fastrepl.Dataset.from_dict({self._output_feature: data})
