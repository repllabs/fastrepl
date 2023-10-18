from typing import Optional, Dict, List, Any
from collections import defaultdict
from datetime import datetime

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, CBEvent, EventPayload

from fastrepl import Dataset


class FastREPLCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        dataset: Dataset,
        event_starts_to_ignore: List[CBEventType] = [],
        event_ends_to_ignore: List[CBEventType] = [],
    ) -> None:
        self.ds = dataset
        self._active = True
        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)

        super().__init__(
            event_starts_to_ignore,
            event_ends_to_ignore,
        )

    def get_ds(self):
        return self.ds

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any
    ) -> str:
        if event_type == CBEventType.QUERY and payload is not None:
            event = CBEvent(event_type, payload=payload, id_=event_id)
            self._event_pairs_by_id[event.id_].append(event)

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        if event_type == CBEventType.QUERY or event_type == CBEventType.RETRIEVE:
            event = CBEvent(event_type, payload=payload, id_=event_id)
            self._event_pairs_by_id[event.id_].append(event)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._start_time = (
            datetime.now()
        )  # TODO: not thread safe, need to do this on query start.

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if trace_map is None:
            return

        ids = []
        for top_id in trace_map["root"]:
            ids.append(top_id)
            for id in trace_map[top_id]:
                ids.append(id)

        data: Dict[str, Any] = {}

        for id in ids:
            events = self._event_pairs_by_id.pop(id, [])
            if len(events) == 0:
                continue

            if len(events) == 1:
                data["contexts"] = [
                    str(n.text) for n in events[0].payload.get(EventPayload.NODES, [])  # type: ignore[union-attr]
                ]
            elif len(events) == 2:
                data["question"] = events[0].payload.get(EventPayload.QUERY_STR)  # type: ignore[union-attr]
                data["answer"] = str(events[1].payload.get(EventPayload.RESPONSE))  # type: ignore[union-attr]

        if self.active():
            self.ds.add_row(data)

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def active(self) -> bool:
        return self._active
