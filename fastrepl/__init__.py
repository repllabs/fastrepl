from typing import Optional

from fastrepl.version import __version__

telemetry: bool = True
api_base: Optional[str] = "https://yujonglee--fastrepl-api.modal.run"  # TODO
api_key: Optional[str] = None

from fastrepl.eval import *
from fastrepl.gen import *
from fastrepl.analyze import Analyzer
from fastrepl.dataset import Dataset

from fastrepl.utils import DEBUG

from fastrepl.runner import *

from fastrepl.pytest_plugin import TestReport

from fastrepl.telemetry import _import_package

_import_package()
