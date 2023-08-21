from fastrepl.eval.model import (
    LLMClassifier,
    LLMChainOfThoughtClassifier,
)


class TestClassifier:
    def test_single_classifier(self):
        eval = LLMClassifier(
            model="gpt-3.5-turbo",
            labels={"A": "POSITIVE", "B": "NEGATIVE"},
        )

        assert eval.compute("What a great day!") == "POSITIVE"
        assert eval.compute("What a bad day!") == "NEGATIVE"

    def test_cot_and_classify(self):
        eval = LLMChainOfThoughtClassifier(
            model="gpt-4",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels={"A": "POSITIVE", "B": "NEGATIVE"},
        )
        assert eval.compute("What a great day!") == "NEGATIVE"
        assert eval.compute("What a bad day!") == "POSITIVE"
