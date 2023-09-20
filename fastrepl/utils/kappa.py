from typing import List, Any, cast

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa, aggregate_raters


def kappa(predictions: List[List[Any]]) -> float:
    if len(predictions) < 2:
        raise ValueError
    if any(len(ps) == 0 for ps in predictions):
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

    if len(predictions) == 2:
        return _cohens_kappa(predictions[0], predictions[1])
    return _fleiss_kappa(predictions)


def _cohens_kappa(a: List[Any], b: List[Any]) -> float:
    return cohens_kappa(
        table=confusion_matrix(a, b),
        return_results=False,
    )


def _fleiss_kappa(predictions: List[List[Any]]) -> float:
    input = list(zip(*predictions))  # transpose
    table, _ = aggregate_raters(input)
    return fleiss_kappa(table)
