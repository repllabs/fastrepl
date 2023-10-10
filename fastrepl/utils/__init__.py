from fastrepl.utils.id import get_cuid
from fastrepl.utils.env import loadenv, setenv, getenv
from fastrepl.utils.iterator import pairwise
from fastrepl.utils.data_structure import OrderedSet, HistoryDict
from fastrepl.utils.ensure import ensure
from fastrepl.utils.context import LocalContext, Variable
from fastrepl.utils.prompt import prompt
from fastrepl.utils.print import console, suppress
from fastrepl.utils.debug import debug, DEBUG
from fastrepl.utils.string import truncate, to_number
from fastrepl.utils.kappa import kappa
from fastrepl.utils.llm import (
    raise_openai_exception_for_retry,
    RetryConstantException,
    RetryExpoException,
)
from fastrepl.utils.number import map_number_range
