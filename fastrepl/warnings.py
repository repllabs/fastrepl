import warnings
from typing import cast


def warning_formatter(message, category, filename, lineno, line=None):
    if not hasattr(category, "fastrepl"):  # NOTE: this is built-in warning formatting
        msg = warnings.WarningMessage(message, category, filename, lineno, None, line)
        return warnings._formatwarnmsg_impl(msg)

    category = cast(Warning, category)

    if str(message) == "":
        return f"{filename}:{lineno}: {category.__name__} | {category.doc_url()}\n"

    return (
        f"{filename}:{lineno}: {category.__name__}: {message} | {category.doc_url()}\n"
    )


warnings.formatwarning = warning_formatter


def warn(message="", category=Warning):
    warnings.warn(message, category)


class Warning(UserWarning):
    @staticmethod
    def fastrepl() -> bool:
        return True

    @staticmethod
    def doc_url() -> str:
        raise NotImplementedError


class VerbosityBias(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com"


class IncompletePrediction(Warning):
    @staticmethod
    def doc_url() -> str:
        return "https://docs.fastrepl.com"
