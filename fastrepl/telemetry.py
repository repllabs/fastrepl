import threading


def _get_id():
    import os

    file_path = os.path.expanduser("~/.cache/fastrepl/id.txt")
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        import uuid

        id = uuid.uuid4().hex
        with open(file_path, "w") as file:
            file.write(id)
            return id


def _check_telemetry():
    import os
    import fastrepl

    if os.environ.get("FASTREPL_TELEMETRY", "1") == "0":
        return False

    return fastrepl.telemetry


def _is_colab():
    try:
        import google.colab  # type: ignore

        return True
    except:
        return False


def _import_package():
    if not _check_telemetry():
        return

    import fastrepl
    import httpx

    def send():
        httpx.post(
            f"{fastrepl.api_base}/log",
            json={
                "event": "import",
                "user_id": _get_id(),
                "colab": _is_colab(),
                "version": fastrepl.__version__,
            },
        )

    threading.Thread(target=send).start()
