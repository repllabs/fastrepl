from fastrepl.eval.model import (
    LLMClassifier,
    LLMChainOfThought,
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

    def test_cot_with_classifier(self):
        input = "What a great day!"
        labels = {"A": "POSITIVE", "B": "NEGATIVE"}

        thinker = LLMChainOfThought(
            model="gpt-3.5-turbo",
            labels=labels,
            context="You will get a input text by a liar. Take it as the opposite.",
        )

        thought = thinker.compute(input)

        classifier = LLMClassifier(
            model="gpt-4",
            context=thought,
            labels=labels,
        )

        assert classifier.compute(input) == "NEGATIVE"

    def test_cot_and_classify(self):
        eval = LLMChainOfThoughtClassifier(
            model="gpt-4",
            context="You will get a input text by a liar. Take it as the opposite.",
            labels={"A": "POSITIVE", "B": "NEGATIVE"},
        )
        assert eval.compute("What a great day!") == "NEGATIVE"
        assert eval.compute("What a bad day!") == "POSITIVE"
