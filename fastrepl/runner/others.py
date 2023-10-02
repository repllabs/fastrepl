from typing import Callable, Optional, Iterable, Mapping, List, Any, cast

from concurrent.futures import ThreadPoolExecutor, Future
from rich.progress import Progress

import fastrepl
from fastrepl.utils import getenv, console

NUM_THREADS = getenv("NUM_THREADS", 8)


class LocalCustomRunner:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def run(
        self,
        args_list: Optional[List[Iterable[Any]]] = None,
        kwds_list: Optional[List[Mapping[str, Any]]] = None,
        output_feature="sample",
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

            with ThreadPoolExecutor(min(NUM_THREADS, len(args_list))) as executor:
                futures: List[Future] = []

                for args, kwds in zip(args_list, kwds_list):
                    future = executor.submit(self._fn, *args, **kwds)
                    future.add_done_callback(cb)
                    futures.append(future)

                samples = [future.result() for future in futures]
                return fastrepl.Dataset.from_dict({output_feature: samples})
