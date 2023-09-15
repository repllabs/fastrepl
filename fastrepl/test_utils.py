import json


class TestReport:
    @staticmethod
    def add(data) -> None:
        s = json.dumps(data)
        # NOTE: This will be later parsed by fastrepl
        print(f"__FASTREPL_START_{s}_FASTREPL_END__")
