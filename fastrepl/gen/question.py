from fastrepl.gen.base import BaseGenerator


class QuestionGenerator(BaseGenerator):
    def __init__(self, source: str) -> None:
        self.kind = "question"
        self.source = source
