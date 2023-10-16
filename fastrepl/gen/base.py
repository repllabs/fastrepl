from dataclasses import dataclass


@dataclass
class BaseGenerator:
    kind: str
    source: str
