from typing import Dict
import functools

from rich.prompt import Prompt

from fastrepl.eval.human.base import BaseHumanEval


class HumanClassifier(BaseHumanEval):
    def __init__(
        self,
        labels: Dict[str, str],
        instruction: str = "Classify the following sample",
        prompt_template="[bright_magenta]{instruction}:[/bright_magenta]\n\n{sample}\n\n",
    ) -> None:
        self.labels = labels
        self.render_prompt = functools.partial(
            prompt_template.format, instruction=instruction
        )

    def compute(self, sample: str, context="") -> str:
        prompt = self.render_prompt(sample=sample)
        # TODO: Render descriptions
        choices = list(self.labels.keys())

        if context == "":
            return Prompt.ask(prompt, choices=choices)
        else:
            return Prompt.ask(prompt, choices=choices, default=context)
