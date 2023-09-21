from typing import List, Any, cast

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa, aggregate_raters


def kappa(predictions: List[List[Any]]) -> float:
    assert isinstance(predictions[0], list)

    num_raters = len(predictions[0])
    if num_raters < 2:
        raise ValueError

    # TODO: hacky none-handling
    if isinstance(predictions[0][0], str):
        for ps in predictions:
            ps = ["" if p is None else p for p in ps]

        le = LabelEncoder()
        le.fit(list(set([p for ps in predictions for p in ps])))

        predictions = [le.transform(p) for p in predictions]
    else:
        predictions = [[-1 if p is None else p for p in ps] for ps in predictions]

    kappa_fn = _cohens_kappa if num_raters == 2 else _fleiss_kappa

    return kappa_fn(predictions)


def _cohens_kappa(predictions: List[List[Any]]) -> float:
    predictions = list(zip(*predictions))  # transpose

    return cohens_kappa(
        table=confusion_matrix(*predictions),
        return_results=False,
    )


def _fleiss_kappa(predictions: List[List[Any]]) -> float:
    table, _ = aggregate_raters(predictions)
    return fleiss_kappa(table)
