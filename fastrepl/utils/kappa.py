from typing import List, Any

from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa


def kappa(*predicions: List[Any]):
    if len(predicions) < 2:
        raise ValueError
    if len(predicions) > 2:
        raise NotImplementedError

    table = confusion_matrix(predicions[0], predicions[1])

    return cohens_kappa(table=table, return_results=False)
