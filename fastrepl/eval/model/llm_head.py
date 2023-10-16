import random
import functools
import itertools

from abc import abstractmethod
from typing import Optional, Union, Tuple, Iterable, TypedDict, List, Dict, cast
from typing_extensions import Unpack, NotRequired

import fastrepl.llm as llm
from fastrepl.utils import prompt, to_number, map_number_range
from fastrepl.eval.base import BaseSimpleEvalNode

from fastrepl.warnings import (
    warn,
    VerbosityBiasWarning,
    InvalidPredictionWarning,
)
from fastrepl.eval.model.utils import (
    logit_bias_from,
    mappings_from_labels,
    next_mappings_for_consensus,
    check_length_inbalance,
    PositionDebiasStrategy,
)


class LLMEvaluationHeadParams(TypedDict):
    context: str
    options: NotRequired[Iterable[str]]  # TODO: It is actually required.
    model: NotRequired[str]
    rg: NotRequired[random.Random]
    references: NotRequired[List[Tuple[str, str]]]


class LLMEvaluationHead(BaseSimpleEvalNode):
    def __init__(self, **kwargs: Unpack[LLMEvaluationHeadParams]) -> None:
        self.context = kwargs["context"]
        self.options = kwargs["options"]
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.rg = kwargs.get("rg", random.Random(42))
        self.references = kwargs.get("references", [])

    @abstractmethod
    def system_message(self, sample: str, context: str) -> Dict[str, str]:
        ...

    @abstractmethod
    def final_message(self, sample: str, context: str) -> Dict[str, str]:
        ...

    def reference_messages(
        self,
        references: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        return list(
            itertools.chain.from_iterable(
                (
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": output},
                )
                for input, output in references
            )
        )

    def messages(self, sample: str) -> List[Dict[str, str]]:
        system_message = self.system_message(sample, self.context)
        reference_messages = self.reference_messages(
            self.rg.sample(self.references, len(self.references))
        )
        final_message = self.final_message(sample, self.context)

        return [system_message, *reference_messages, final_message]

    def completion(self, sample: str) -> Optional[str]:
        logit_bias = logit_bias_from(self.model, [str(i) for i in self.options])
        max_tokens = 1 if logit_bias != {} else 2

        return llm.completion(
            model=self.model,
            messages=self.messages(sample),
            max_tokens=max_tokens,
            logit_bias=logit_bias,
        )["choices"][0]["message"]["content"]

    # NOTE: It is safe to return NONE, since metric will skip prediction-reference pair if prediction is NONE
    def run(self, *, sample: str) -> Optional[Union[str, float]]:
        result = self.completion(sample)
        if result is None:
            return None

        # we can get ' A' instead of 'A', which is still a single token.
        result = result.strip()

        # NOTE: Although we use max_token=1 and logit_bias, we still can get something different.
        # This is because 1. some LLM provider does not have logit_bias option
        # 2. for Cohere, max logit_bias value(=10) is not enough to force the model. (Not sure why.)
        if result not in self.options:
            warn(InvalidPredictionWarning, context=f"{result!r} not in {self.options}.")
            return None

        return result


class LLMClassificationHead(LLMEvaluationHead):
    def __init__(
        self,
        labels: Dict[str, str],
        position_debias_strategy: PositionDebiasStrategy = "shuffle",
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        if check_length_inbalance(labels.values()):  # pragma: no cover
            warn(VerbosityBiasWarning)

        self.labels = labels
        self.mapping = mappings_from_labels(labels)
        self.position_debias_strategy: PositionDebiasStrategy = position_debias_strategy

        kwargs.update({"options": [m.token for m in self.mapping]})
        super().__init__(**kwargs)

    def system_message(self, sample: str, context: str) -> Dict[str, str]:
        @prompt
        def p(context, labels, label_keys):
            """You are master of classification who can classify any text according to the user's instructions.
            {{context}}

            These are the labels you can use:
            {{labels}}

            Only output one of these label keys:
            {{label_keys}}"""

        return {
            "role": "system",
            "content": p(
                context=context,
                labels="\n".join(f"{m.token}: {m.description}" for m in self.mapping),
                label_keys=", ".join(m.token for m in self.mapping),
            ),
        }

    def reference_messages(
        self, references: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        def label2token(text: str) -> str:
            return functools.reduce(
                lambda t, m: t.replace(m.label, m.token), self.mapping, text
            )

        return super().reference_messages(
            [(input, label2token(output)) for input, output in references]
        )

    def final_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "user", "content": sample}

    def _compute(self, sample: str) -> Optional[str]:
        if self.position_debias_strategy == "shuffle":
            self.mapping = mappings_from_labels(self.labels, rg=self.rg)
            return cast(str, super().run(sample=sample))

        if self.position_debias_strategy == "consensus":
            initial_result = cast(str, super().run(sample=sample))
            if initial_result is None:
                return None

            next_mapping = next_mappings_for_consensus(self.mapping, initial_result)
            if next_mapping is None:
                return initial_result

            self.mapping = next_mapping
            next_result = super().run(sample=sample)
            if next_result is None:
                return None

            return initial_result if initial_result == next_result else None

    def run(self, *, sample: str) -> Optional[str]:
        token = self._compute(sample)
        if token is None:
            return None

        return next(m.label for m in self.mapping if m.token == token)


class LLMGradingHead(LLMEvaluationHead):
    def __init__(
        self,
        number_from: int,
        number_to: int,
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        if number_to < 0 or number_from < 0 or number_to < number_from:
            raise ValueError

        if number_to < 10 and number_from < 10 and number_to - number_from < 4:
            # We don't need mapping
            self.from_min, self.from_max = number_from, number_to
            self.to_min, self.to_max = number_from, number_to
        else:
            # We grade within [1, 5], and map the result to [number_from, number_to].
            self.from_min, self.from_max = 1, 5
            self.to_min, self.to_max = number_from, number_to

        options = [str(i) for i in range(self.from_min, self.from_max + 1)]
        kwargs.update({"options": options})
        super().__init__(**kwargs)

    def system_message(self, sample: str, context: str) -> Dict[str, str]:
        @prompt
        def p(context, min, max):
            """You are master of grading who can grade any text according to the context information that the user gives.

            Context: {{context}}

            Now, you will receive a text to grade. Output a single integer number from {{min}} to {{max}}.
            """

        return {"role": "system", "content": p(context, self.from_min, self.from_max)}

    def reference_messages(
        self,
        references: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        msgs = []
        for input, output in references:
            score = float(str(output).strip())
            score = map_number_range(
                score, self.to_min, self.to_max, self.from_min, self.from_max
            )
            score = round(score)
            msgs.append({"role": "user", "content": input})
            msgs.append({"role": "assistant", "content": str(score)})
        return msgs

    def final_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "user", "content": sample}

    def run(self, *, sample: str) -> Optional[float]:
        completion = self.completion(sample)

        result = to_number(completion)
        if result is None:
            warn(
                InvalidPredictionWarning,
                context=f"Unable to convert {completion!r} to number",
            )
            return None

        result = map_number_range(
            result, self.from_min, self.from_max, self.to_min, self.to_max
        )

        if result < self.to_min or result > self.to_max:
            closest = self.to_min if result < self.to_min else self.to_max

            warn(
                InvalidPredictionWarning,
                context=f"{result!r} is not in range [{self.to_min}, {self.to_max}]. Using {closest!r} instead.",
            )
            return closest

        return result
