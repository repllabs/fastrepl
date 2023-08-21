import random
from typing import Tuple, Dict, List

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels


class LLMChainOfThoughtClassifier(BaseModelEval):
    __slots__ = ("criteria", "labels", "references", "rg", "model")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.criteria = context
        self.labels = labels
        self.references = references
        self.model = model
        self.rg = rg

    def compute(self, sample: str) -> str:
        references = self.rg.sample(self.references, len(self.references))

        system_content = f"""If user gave you the text, do step by step thinking and classify it.

When do step-by-step thinking, you must consider the following:
{self.criteria}

Step-by-step thinking should use less than 30 words.

For classification, use the following labels(<LABEL>:<DESCRIPTION>):
{render_labels(self.labels)}

Output only this format: ### Thoghts: <STEP_BY_STEP_THOUGHTS>\n### Label: <LABEL>"""

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

        result = (
            completion(
                self.model,
                messages=messages,
                max_tokens=200,
            )
            .choices[0]
            .message.content
        )
        # TODO: validate
        return self.labels.get(result[-1], "UNKNOWN")
