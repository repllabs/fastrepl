import random
import warnings
from typing import Optional, Tuple, Literal, Dict, List

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference
from fastrepl.eval.model.utils import (
    logit_bias_from_labels,
    mappings_from_labels,
    next_mappings_for_consensus,
    LabelMapping,
)


@prompt
def system_prompt(context, labels, label_keys):
    """You are master of classification who can classify any text according to the user's instructions.
    {{context}}

    These are the labels you can use:
    {{labels}}

    Only output one of these label keys:
    {{label_keys}}"""


@prompt
def final_message_prompt(sample, context=""):
    """{% if context != '' %}
    Info about the text: {{ context }}
    {% endif %}
    Text to think about: {{ sample }}"""


class LLMClassifier(BaseEvalWithoutReference):
    __slots__ = ("model", "mapping", "rg", "references", "system")

    def __init__(
        self,
        labels: Dict[str, str],
        context: str = "",
        model: SUPPORTED_MODELS = "gpt-3.5-turbo",
        rg=random.Random(42),
        references: List[Tuple[str, str]] = [],
        position_debias_strategy: Literal["shuffle", "consensus"] = "shuffle",
    ) -> None:
        self.labels = labels
        self.global_context = context
        self.model = model
        self.rg = rg
        self.references = references
        self.position_debias_strategy: Literal[
            "shuffle", "consensus"
        ] = position_debias_strategy

    def _compute(
        self,
        sample: str,
        context: str,
        mappings: List[LabelMapping],
        references: List[Tuple[str, str]],
    ) -> Optional[LabelMapping]:
        instruction = system_prompt(
            context=self.global_context,
            labels="\n".join(f"{m.token}: {m.description}" for m in mappings),
            label_keys=", ".join(m.token for m in mappings),
        )

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
            logit_bias=logit_bias_from_labels(
                self.model, set(m.token for m in mappings)
            ),
        )["choices"][0]["message"]["content"]

        for m in mappings:
            if m.token == result:
                return m

        warnings.warn(f"classification result not in mapping: {result!r}")
        return None

    def compute(self, sample: str, context="") -> Optional[str]:
        mappings = mappings_from_labels(self.labels, rg=self.rg)
        references = self.rg.sample(self.references, len(self.references))

        mapping1 = self._compute(sample, context, mappings, references)
        if mapping1 is None:
            return None

        if self.position_debias_strategy == "shuffle":
            return mapping1.label

        next_mappings = next_mappings_for_consensus(mappings, mapping1)
        if next_mappings is None:
            return mapping1.label

        mapping2 = self._compute(sample, context, next_mappings, references)
        if mapping2 is None:
            return None

        if mapping1.label == mapping2.label:
            return mapping1.label
        else:
            return None

    def is_interactive(self) -> bool:
        return False
