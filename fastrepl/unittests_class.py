import unittest

from fastrepl.utils import getenv

SKIP_FASTREPL = getenv("SKIP_FASTREPL", True)


@unittest.skipIf(SKIP_FASTREPL, "fastrepl is not enabled")
class FastreplTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_base = getenv("LITELLM_PROXY_API_BASE", "")
        api_key = getenv("LITELLM_PROXY_API_KEY", "")

        is_proxy = api_base != "" and api_key != ""
        print(f"Using proxy: {is_proxy}")
