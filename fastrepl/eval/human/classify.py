from typing import Optional, TextIO, Dict
import functools

from fastrepl.eval.human.base import BaseHumanEval


class HumanClassifierRich(BaseHumanEval):
    def __init__(
        self,
        labels: Dict[str, str],
        instruction: str = "Classify the following sample",
        prompt_template="[bright_magenta]{instruction}:[/bright_magenta]\n\n{sample}\n\n",
        stream=Optional[TextIO],
    ) -> None:
        self.labels = labels
        self.render_prompt = functools.partial(
            prompt_template.format, instruction=instruction
        )
        self.stream = stream

    def compute(self, sample: str, context="") -> str:
        from rich.prompt import Prompt

        prompt = self.render_prompt(sample=sample)
        choices = list(self.labels.keys())  # TODO: Render descriptions

        if context == "":
            return Prompt.ask(
                prompt,
                choices=choices,
                stream=self.stream,
            )
        else:
            return Prompt.ask(
                prompt,
                choices=choices,
                default=context,
                stream=self.stream,
            )
