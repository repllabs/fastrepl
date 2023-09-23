from typing import Optional

api_base: Optional[str] = "https://yujonglee--fastrepl-api.modal.run"  # TODO
api_key: Optional[str] = None

from fastrepl.eval import (
    load_metric,
    HumanClassifierRich,
    LLMGradingHead,
    LLMClassificationHead,
    LLMGradingHeadCOT,
    LLMClassificationHeadCOT,
    Evaluator,
    SimpleEvaluator,
    RAGEvaluator,
)
from fastrepl.analyze import Analyzer
from fastrepl.dataset import Dataset

from fastrepl.utils import DEBUG

from fastrepl.runner import (
    LocalRunner,
    RemoteRunner,
)

from fastrepl.pytest_plugin import TestReport
