# All comments are in English.
from __future__ import annotations
from typing import Any, Callable, Dict

ClientFactory = Callable[[Dict[str, Any]], Any]
AdapterFactory = Callable[..., Any]

class _Registry:
    def __init__(self) -> None:
        self._clients: Dict[str, ClientFactory] = {}
        self._adapters: Dict[str, AdapterFactory] = {}

    # --- imperative registration (still available) ---
    def register_client(self, backend: str, factory: ClientFactory) -> None:
        self._clients[backend.lower()] = factory

    def register_adapter(self, backend: str, factory: AdapterFactory) -> None:
        self._adapters[backend.lower()] = factory

    def build_client(self, backend: str, cfg: Dict[str, Any]):
        b = backend.lower()
        if b not in self._clients:
            raise ValueError(f"No client factory registered for backend '{backend}'.")
        return self._clients[b](cfg)

    def build_adapter(self, backend: str, **kwargs):
        b = backend.lower()
        if b not in self._adapters:
            raise ValueError(f"No adapter factory registered for backend '{backend}'.")
        return self._adapters[b](**kwargs)

REGISTRY = _Registry()


# --- decorator-based registration ---
def register_client(*names: str):
    """Decorator to register a client factory function for one or more backend names."""
    def decorator(fn: ClientFactory) -> ClientFactory:
        for name in names:
            REGISTRY.register_client(name, fn)
        return fn
    return decorator


def register_adapter(*names: str):
    """Decorator to register an adapter class for one or more backend names."""
    def decorator(cls: AdapterFactory) -> AdapterFactory:
        for name in names:
            REGISTRY.register_adapter(name, lambda cls=cls, **kw: cls(**kw))
        return cls
    return decorator
