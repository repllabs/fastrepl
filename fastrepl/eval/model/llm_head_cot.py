from typing import Optional, Dict

import fastrepl.llm as llm
from fastrepl.utils import prompt
from fastrepl.eval.model import LLMClassificationHead, LLMGradingHead


class LLMClassificationHeadCOT(LLMClassificationHead):
    def system_message(self, sample: str, context: str) -> Dict[str, str]:
        @prompt
        def p(context, labels):
            """You are master of classification who can classify any text according to the user's instructions.
            When user give you the text to classify, you do step-by-step thinking within 3 sentences and give a final result.

            When doing step-by-step thinking, you must consider the following:
            {{ context }}

            These are the labels(KEY: DESCRIPTION) you can use:
            {{labels}}

            Your response must strictly follow this format:
            ### Thoughts
            <STEP_BY_STEP_THOUGHTS>
            ### Result
            <SINGLE_LABEL_KEY>"""

        return {
            "role": "system",
            "content": p(
                context=context,
                labels="\n".join(f"{m.token}: {m.description}" for m in self.mapping),
            ),
        }

    def completion(self, sample: str) -> Optional[str]:
        prediction: str = llm.completion(
            model=self.model, messages=self.messages(sample)
        )["choices"][0]["message"]["content"]
        prediction = prediction.split("### Result")[-1].strip()

        # TODO
        for m in self.mapping:
            for delimiter in [":", "\n"]:
                if prediction.startswith(f"{m.token}{delimiter}"):
                    return m.token

        return prediction


class LLMGradingHeadCOT(LLMGradingHead):
    def system_message(self, sample: str, context: str) -> Dict[str, str]:
        @prompt
        def p(context):
            """You are master of grading who can grade any text according to the user's instructions.
            When user give you the text to grade, you do step-by-step thinking within 5 sentences and give a final result.

            When doing step-by-step thinking, you must consider the following:
            {{ context }}

            Your response must strictly follow this format:
            ### Thoughts
            <STEP_BY_STEP_THOUGHTS>
            ### Result
            <NUMBER>"""

        return {"role": "system", "content": p(context)}

    def completion(self, sample: str) -> Optional[str]:
        result = llm.completion(model=self.model, messages=self.messages(sample))[
            "choices"
        ][0]["message"]["content"]

        return result.split("### Result")[-1].strip()
