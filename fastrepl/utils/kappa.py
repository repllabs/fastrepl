from typing import List, Any

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa, aggregate_raters


def kappa(*predictions: List[Any]) -> float:
    if len(predictions) < 2:
        raise ValueError
    if any(len(ps) == 0 for ps in predictions):
        raise ValueError

    if len(predictions) == 2:
        return _cohens_kappa(predictions[0], predictions[1])
    return _fleiss_kappa(*predictions)


def _cohens_kappa(pred_a: List[Any], pred_b: List[Any]) -> float:
    if all(isinstance(p, str) for p in pred_a + pred_b):
        # TODO: workaround for None
        a = ["" if p is None else p for p in pred_a]
        b = ["" if p is None else p for p in pred_b]

        le = LabelEncoder()
        le.fit(list(set(a + b)))

        a, b = le.transform(a), le.transform(b)
    else:
        # TODO: workaround for None
        a = [-1 if p is None else p for p in pred_a]
        b = [-1 if p is None else p for p in pred_b]

    return cohens_kappa(table=confusion_matrix(a, b), return_results=False)


def _fleiss_kappa(*predictions: List[Any]) -> float:
    input = list(zip(*predictions))  # transpose

    # TODO: NONE handling, maybe we should add that to _kappa

    table, _ = aggregate_raters(input)
    return fleiss_kappa(table)
