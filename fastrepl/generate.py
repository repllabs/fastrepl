from dataclasses import dataclass


@dataclass
class Generator:
    kind: str
    source: str


class QuestionGenerator(Generator):
    def __init__(self, source: str) -> None:
        self.kind = "question"
        self.source = source
