from rich.pretty import pprint

from fastrepl.utils import Variable, console


DEBUG = Variable("DEBUG", 0)


def debug(input, before="", after="") -> None:
    if DEBUG < 1:
        return None

    if before != "":
        console.rule(before, align="left", style="white")

    if DEBUG > 1:
        pprint(input, expand_all=True)
    else:
        pprint(input, expand_all=True, max_string=110)

    if after != "":
        console.rule(after, align="left", style="white")
