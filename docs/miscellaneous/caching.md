# Caching

Currently, we use in-memory caching using [model's name and prompt as a key](https://docs.litellm.ai/docs/caching).

Eventually, we will have our own implementation of disk-based caching, which will save more API costs and time.
