from typing import List, Any

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import cohens_kappa


def kappa(*predicions: List[Any]):
    if len(predicions) < 2:
        raise ValueError
    if len(predicions) > 2:
        raise NotImplementedError

    if isinstance(predicions[0][0], str):
        le = LabelEncoder()
        labels = list(set(predicions[0] + predicions[1]))
        le.fit(labels)

        a = le.transform(predicions[0])
        b = le.transform(predicions[1])
    else:
        a = predicions[0]
        b = predicions[1]

    return cohens_kappa(table=confusion_matrix(a, b), return_results=False)
