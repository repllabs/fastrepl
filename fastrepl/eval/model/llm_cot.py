import random
from typing import Tuple, List, Dict

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels


class LLMChainOfThought(BaseModelEval):
    __slots__ = ("context", "labels", "references", "rg", "model", "length")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
        length=2,
    ) -> None:
        self.model = model
        self.context = context
        self.labels = labels
        self.references = references
        self.rg = rg
        self.length = length

    def compute(self, sample: str, context="") -> str:
        references = self.rg.sample(self.references, len(self.references))

        system_content = f"""If user gave you the text, do step by step thinking that is needed to classify it.
Use less than 50 words.

These are the labels that will be used later to classify the text:
{render_labels(self.labels)}

When do step-by-step thinking, you must consider the following:
{self.context}

### Thoghts:"""

        messages = [
            {
                "role": "system",
                "content": system_content,
            }
        ]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": sample})

        result = completion(self.model, messages=messages).choices[0].message.content
        return result
