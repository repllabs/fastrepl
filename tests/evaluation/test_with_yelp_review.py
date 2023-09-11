import pytest
from datasets import Dataset, load_dataset

import fastrepl
from fastrepl.utils import number


def eval_name(evaluator: str, model: str) -> str:
    return f"fastrepl_yelp_review_{evaluator}_{model}"


def grade2number(example):
    example["prediction"] = number(example["prediction"])
    return example


@pytest.fixture
def dataset() -> Dataset:
    dataset = load_dataset("yelp_review_full", split="test")
    dataset = dataset.shuffle(seed=8)
    dataset = dataset.select(range(30))
    dataset = dataset.rename_column("text", "input")
    dataset = dataset.map(
        lambda row: {"reference": row["label"] + 1, "input": row["input"]},
        remove_columns=["label"],
    )
    return dataset


@pytest.mark.parametrize(
    "model, references",
    [
        (
            "gpt-3.5-turbo",
            [],
        ),
        (
            "togethercomputer/llama-2-70b-chat",
            [
                ("Text to grade: this place is nice!", "4"),
                ("Text to grade: this place is so bad", "1"),
            ],
        ),
    ],
)
@pytest.mark.fastrepl
def test_llm_grading_head(dataset, model, references, report: fastrepl.TestReport):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMGradingHead(
                model=model,
                context="You will get a input text from Yelp review. Grade user's satisfaction from 1 to 5.",
                number_from=1,
                number_to=5,
                references=references,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(grade2number)

    predictions = result["prediction"]
    references = result["reference"]

    # fmt: off
    accuracy = fastrepl.load_metric("accuracy").compute(predictions, references)["accuracy"]
    mse = fastrepl.load_metric("mse").compute(predictions, references)["mse"]
    mae = fastrepl.load_metric("mae").compute(predictions, references)["mae"]
    # fmt: on

    report.add(
        {
            "eval": "LLMGradingHead",
            "model": model,
            "accuracy": accuracy,
            "mse": mse,
            "mae": mae,
        }
    )
    assert accuracy > 0.09
    assert mse < 6
    assert mae < 3


@pytest.mark.parametrize(
    "model",
    [
        ("gpt-3.5-turbo"),
    ],
)
@pytest.mark.fastrepl
def test_grading_head_cot(dataset, model, report: fastrepl.TestReport):
    eval = fastrepl.Evaluator(
        pipeline=[
            fastrepl.LLMGradingHeadCOT(
                model=model,
                context="You will get a input text from Yelp review. Grade user's satisfaction in integer from 1 to 5.",
                number_from=1,
                number_to=5,
            )
        ]
    )

    result = fastrepl.LocalRunner(evaluator=eval, dataset=dataset).run()
    result = result.map(grade2number)

    predictions = result["prediction"]
    references = result["reference"]

    # fmt: off
    accuracy = fastrepl.load_metric("accuracy").compute(predictions, references)["accuracy"]
    mse = fastrepl.load_metric("mse").compute(predictions, references)["mse"]
    mae = fastrepl.load_metric("mae").compute(predictions, references)["mae"]
    # fmt: on

    report.add(
        {
            "eval": "LLMGradingHeadCOT",
            "model": model,
            "accuracy": accuracy,
            "mse": mse,
            "mae": mae,
        }
    )
    assert accuracy > 0.09
    assert mse < 6
    assert mae < 3
