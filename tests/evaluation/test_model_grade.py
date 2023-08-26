import pytest
from datasets import Dataset

from fastrepl.eval.metric import load_metric
from fastrepl.eval.model import (
    LLMClassifier,
    LLMChainOfThought,
    LLMChainOfThoughtClassifier,
)


class TestClassifier:
    @pytest.mark.fastrepl
    def test_single_classifier(self):
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }
        eval = LLMClassifier(
            model="gpt-3.5-turbo",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels=labels,
        )

        tc = Dataset.from_dict(
            {
                "input": [
                    "What a great day!",
                    "What a bad day!",
                    "I am so happy.",
                    "I am so sad.",
                ],
                "reference": [
                    "POSITIVE",
                    "NEGATIVE",
                    "NEGATIVE",
                    "POSITIVE",
                ],
            }
        )

        predictions, references = [eval.compute(input) for input in tc["input"]]
        references = tc["reference"]

        assert load_metric("accuracy").compute(predictions, references) > 0.5

    @pytest.mark.fastrepl
    def test_cot_with_classifier(self):
        input = "What a great day!"
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }

        pipeline = [
            LLMChainOfThought(
                model="gpt-3.5-turbo",
                labels=labels,
                context="You will get a input text by a liar. Take it as the opposite.",
            ),
            LLMClassifier(
                model="gpt-4",
                labels=labels,
            ),
        ]

        tc = Dataset.from_dict(
            {
                "input": [
                    "What a great day!",
                    "What a bad day!",
                    "I am so happy.",
                    "I am so sad.",
                ],
                "reference": [
                    "POSITIVE",
                    "NEGATIVE",
                    "NEGATIVE",
                    "POSITIVE",
                ],
            }
        )

        thought = pipeline[0].compute(input)
        answer = pipeline[1].compute(input, context=thought)
        assert answer in labels.keys()

    @pytest.mark.fastrepl
    def test_cot_and_classify(self):
        labels = {
            "POSITIVE": "Given text is actually positive",
            "NEGATIVE": "Given text is actually negative",
        }
        eval = LLMChainOfThoughtClassifier(
            model="gpt-3.5-turbo",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels=labels,
        )
        assert eval.compute("What a great day! I am so happy.") in labels.keys()
        assert eval.compute("What a bad day! I am so sad.") in labels.keys()
