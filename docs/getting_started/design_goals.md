# Design Goals

## Single Import
`fastrepl` is designed to be used with a single `import fastrepl`. This requires more typing, but it makes it easier to learn and use.

### Non-interactive mode
This is for most use cases. In the normal python soure code, and tests.

```python
import fastrepl
```

## Framework Agnostic
It should be straightforward to use `fastrepl` with any framework or setup.
