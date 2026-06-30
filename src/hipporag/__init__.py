from importlib import import_module

__all__ = ["HippoRAG", "HyperHippoRAG"]


def __getattr__(name):
    if name == "HippoRAG":
        return import_module(".HippoRAG", __name__).HippoRAG
    if name == "HyperHippoRAG":
        return import_module(".HyperHippoRAG", __name__).HyperHippoRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
