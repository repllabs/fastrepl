import random
from typing import Tuple, Dict, List

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.utils import logit_bias_from_labels
from fastrepl.eval.model.base import BaseModelEval


class LLMClassifier(BaseModelEval):
    __slots__ = ("labels", "context", "model", "rg", "references")

    def __init__(
        self,
        labels: Dict[str, str],
        context: str = "",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.labels = labels
        self.context = context
        self.model = model
        self.rg = rg
        self.references = references

    def compute(self, sample: str) -> str:
        references = self.rg.sample(self.references, len(self.references))

        messages = [
            {
                "role": "system",
                "content": f"When I give you a text, please classify it.\nUse these labels: {self.labels}. Output only one of given labels.",
            }
        ]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": sample})

        result = (
            completion(
                self.model,
                messages=messages,
                max_tokens=1,  #  NOTE: when using logit_bias for classification, max_tokens should be 1
                logit_bias=logit_bias_from_labels(
                    self.model,
                    set(self.labels.keys()),
                ),
            )
            .choices[0]
            .message.content
        )
        return self.labels.get(result, "UNKNOWN")
