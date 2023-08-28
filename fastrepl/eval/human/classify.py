from typing import Dict
import functools

from fastrepl.eval.human.base import BaseHumanEval


class HumanClassifierRich(BaseHumanEval):
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
        from rich.prompt import Prompt

        prompt = self.render_prompt(sample=sample)
        choices = list(self.labels.keys())  # TODO: Render descriptions

        if context == "":
            return Prompt.ask(prompt, choices=choices)
        else:
            return Prompt.ask(prompt, choices=choices, default=context)


class HumanClassifierTexual(BaseHumanEval):
    def __init__(
        self,
        labels: Dict[str, str],
        instruction: str = "Classify the following sample",
    ) -> None:
        from textual import on
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Select, RichLog, Button
        from textual.containers import Horizontal

        class ClassificationApp(App[int]):
            TITLE = "FastREPL"
            SUB_TITLE = instruction

            CSS = """
            #log { height: 85%; }
            """

            def __init__(self, text, classes) -> None:
                super().__init__()
                self.text = text
                self.classes = classes
                self.selected = 0

            def compose(self) -> ComposeResult:
                yield Header()
                yield RichLog(id="log", wrap=True)
                with Horizontal():
                    yield Select(
                        value=self.selected,
                        options=((line, i) for i, line in enumerate(self.classes)),
                    )
                    yield Button("Submit", variant="primary")

            def on_ready(self) -> None:
                log = self.query_one(RichLog)
                lines = self.text.split(". ")
                for line in lines:
                    log.write(f"{line}. \n")

            def on_button_pressed(self) -> None:
                self.exit(self.selected)

            @on(Select.Changed)
            def select_changed(self, event: Select.Changed) -> None:
                self.selected = int(str(event.value))

        self.labels = labels
        self.app = ClassificationApp(text="", classes=sorted(labels.keys()))

    def compute(self, sample: str, context="") -> str:
        self.app.text = sample

        result = self.app.run()

        assert result is not None
        return list(self.app.classes)[result]
