from typing import Optional

api_base: Optional[str] = "https://yujonglee--fastrepl-api.modal.run"  # TODO
api_key: Optional[str] = None

from fastrepl.eval import *
from fastrepl.analyze import Analyzer
from fastrepl.generate import Generator, QuestionGenerator

from fastrepl.utils import DEBUG

from fastrepl.runner import local_runner, remote_runner

from fastrepl.pytest_plugin import TestReport
