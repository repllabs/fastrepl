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
        self._id_to_data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._child_to_parent: Dict[str, str] = {}

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
            question = payload.get(EventPayload.QUERY_STR)
            self._id_to_data[event_id]["question"] = question
            self._id_to_data[event_id]["_start"] = datetime.now()

        if event_type == CBEventType.RETRIEVE and payload is not None:
            self._child_to_parent[event_id] = parent_id

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        if event_type == CBEventType.RETRIEVE and payload is not None:
            contexts = [n.text for n in payload.get(EventPayload.NODES, [])]
            parent_id = self._child_to_parent[event_id]
            self._id_to_data[parent_id]["contexts"] = contexts

        if event_type == CBEventType.QUERY and payload is not None:
            answer = str(payload.get(EventPayload.RESPONSE))
            self._id_to_data[event_id]["answer"] = answer
            self._id_to_data[event_id]["_end"] = datetime.now()
            self.update_dataset(event_id)

    def update_dataset(self, event_id: str) -> None:
        data = self._id_to_data.pop(event_id)
        start, end = data.pop("_start"), data.pop("_end")
        elapsed = (end - start).total_seconds()

        row = {k: v for k, v in data.items()}
        row["elapsed"] = elapsed

        self.ds.add_row(row)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def active(self) -> bool:
        return self._active
