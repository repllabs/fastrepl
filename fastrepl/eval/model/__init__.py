from fastrepl.eval.model.llm_head import LLMClassificationHead, LLMGradingHead
from fastrepl.eval.model.llm_head_cot import LLMClassificationHeadCOT, LLMGradingHeadCOT

import sys
from typing import TYPE_CHECKING
from lazy_imports import LazyImporter

from fastrepl.version import __version__

_import_structure = {"eval": [{"model": [{"ragas": ["RAGAS"]}]}]}

if TYPE_CHECKING:
    from fastrepl.eval.model.ragas import RAGAS
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )
