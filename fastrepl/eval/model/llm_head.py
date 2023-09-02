import random
from abc import abstractmethod
from typing import Optional, Tuple, Iterable, TypedDict, List, Dict, cast
from typing_extensions import Unpack, NotRequired

from fastrepl.utils import prompt
from fastrepl.llm import completion, SUPPORTED_MODELS
from fastrepl.eval.base import BaseEvalWithoutReference
from fastrepl.eval.model.utils import logit_bias_from


class LLMEvaluationHeadParams(TypedDict):
    context: str
    options: Iterable[str]
    model: NotRequired[SUPPORTED_MODELS]
    rg: NotRequired[random.Random]
    references: NotRequired[List[Tuple[str, str]]]


class LLMEvaluationHead(BaseEvalWithoutReference):
    def __init__(self, **kwargs: Unpack[LLMEvaluationHeadParams]) -> None:
        self.global_context = kwargs["context"]
        self.options = kwargs["options"]
        self.model = kwargs.get("model", "gpt-3.5-turbo")
        self.rg = kwargs.get("rg", random.Random(42))
        self.references = kwargs.get("references", [])

    @abstractmethod
    def _system_message(self, sample: str, context: str) -> Dict[str, str]:
        ...

    @abstractmethod
    def _final_message(self, sample: str, context: str) -> Dict[str, str]:
        ...

    def _messages(self, sample, context) -> List[Dict[str, str]]:
        references = self.rg.sample(self.references, len(self.references))

        messages = [self._system_message(sample, context)]
        for input, output in references:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
        messages.append(self._final_message(sample, context))
        return messages

    def compute(self, sample: str, context="") -> Optional[str]:
        result = completion(
            model=self.model,
            messages=self._messages(sample, context),
            # NOTE: when using logit_bias for classification, max_tokens should be 1
            max_tokens=1,
            logit_bias=logit_bias_from(self.model, [str(i) for i in self.options]),
        )["choices"][0]["message"]["content"]

        # NOTE: Some LLM provider does not have logit_bias option
        return result if result in self.options else None


class LLMClassificationHead(LLMEvaluationHead):
    def __init__(
        self,
        labels: Dict[str, str],
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        kwargs.update({"options": labels.keys()})  # TODO: Shuffling?
        super().__init__(**kwargs)

    def _system_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "system", "content": sample}

    def _final_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "user", "content": sample}


class LLMGradingHead(LLMEvaluationHead):
    def __init__(
        self,
        number_from: int,
        number_to: int,
        **kwargs: Unpack[LLMEvaluationHeadParams],
    ) -> None:
        kwargs.update({"options": [str(i) for i in range(number_from, number_to + 1)]})
        super().__init__(**kwargs)

    def _system_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "system", "content": sample}

    def _final_message(self, sample: str, context: str) -> Dict[str, str]:
        return {"role": "user", "content": sample}
