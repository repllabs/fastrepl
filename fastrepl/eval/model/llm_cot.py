import random
from typing import Tuple, List, Dict

import outlines.text as text

from fastrepl.run import completion, SUPPORTED_MODELS
from fastrepl.eval.model.base import BaseModelEval
from fastrepl.eval.model.utils import render_labels


@text.prompt
def system_prompt(labels, context):
    """If user gave you the text, do step by step thinking that is needed to classify the given text.
    Use less than 50 words.

    These are the labels that will be used later to classify the text:
    {{labels}}

    When do step-by-step thinking, you must consider the following:
    {{context}}"""


@text.prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to think about: {{ sample }}"""


class LLMChainOfThought(BaseModelEval):
    __slots__ = ("model", "references", "rg", "system_msg")

    def __init__(
        self,
        context: str,
        labels: Dict[str, str],
        previous="",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        self.model = model
        self.references = references
        self.rg = rg
        self.system_msg = {
            "role": "system",
            "content": system_prompt(
                context=context,
                labels=render_labels(labels),
            ),
        }

    def compute(self, sample: str, context="") -> str:
        references = self.rg.sample(self.references, len(self.references))

        messages = [self.system_msg]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": final_message_prompt(sample, context)}
        )

        # fmt: off
        result = completion(
            model=self.model,
            messages=messages, 
        )["choices"][0]["message"]["content"]
        # fmt: on

        return result
