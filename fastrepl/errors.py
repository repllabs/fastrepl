class Error(Exception):
    pass


class DuplicatedKeyError(Error):
    pass


class InvalidStatusError(Error):
    pass
