class Updatable:
    def __init__(self, key: str, value: str):
        self._key, self._value = key, value

    @property
    def value(self) -> str:
        return self._value
