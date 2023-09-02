import random
import warnings
from typing import Optional, Tuple, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference
from fastrepl.eval.model.utils import logit_bias_from


@prompt
def system_prompt(context):
    """You are master of grading who can grade any text according to the user's instructions.
    {{context}}"""


@prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to grade: {{ sample }}"""


class LLMGrader(BaseEvalWithoutReference):
    def __init__(
        self,
        number_from: int,  # TODO: enable providing more info about number
        number_to: int,
        context: str,
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
    ) -> None:
        if number_from > number_to:
            raise ValueError("number_from must be smaller than number_to")
        if number_from < 0 or number_to < 0:
            raise ValueError("number_from and number_to must be positive")

        self.range = range(number_from, number_to + 1)
        self.global_context = context
        self.model = model
        self.rg = rg
        self.references = references

    def _compute(
        self,
        sample: str,
        context: str,
        references: List[Tuple[str, str]],
    ) -> Optional[int]:
        instruction = system_prompt(context=self.global_context)

        messages = [{"role": "system", "content": instruction}]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(
            {"role": "user", "content": final_message_prompt(sample, context)}
        )

        result = completion(
            self.model,
            messages=messages,
            max_tokens=1,  #  NOTE: when using logit_bias for classification, max_tokens should be 1
            logit_bias=logit_bias_from(self.model, [str(i) for i in self.range]),
        )["choices"][0]["message"]["content"]

        try:
            return int(result)
        except ValueError:
            warnings.warn(f"{result!r} not in {self.range}")
            return None

    def compute(self, sample: str, context="") -> Optional[int]:
        references = self.rg.sample(self.references, len(self.references))

        return self._compute(sample, context, references=references)
