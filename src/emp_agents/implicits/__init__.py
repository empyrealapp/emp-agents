from typing import Any, Callable, ParamSpec, TypeVar

from fast_depends import Depends, Provider, inject

from .manager import ImplicitManager
from .models import IgnoreDepends as IgnoreDependsModel

P = ParamSpec("P")
T = TypeVar("T")


def IgnoreDepends(
    dependency: Callable[P, T],
    *,
    use_cache: bool = True,
    cast: bool = True,
) -> Any:
    return IgnoreDependsModel(
        dependency=dependency,
        use_cache=use_cache,
        cast=cast,
    )


def set_implicit(name: str, implicit: Any):
    ImplicitManager.add_implicit(name, implicit)


def lazy_implicit(name: str) -> Callable[P, T] | T:
    return ImplicitManager.lazy_implicit(name)


__all__ = [
    "Depends",
    "ImplicitManager",
    "IgnoreDepends",
    "Provider",
    "inject",
    "set_implicit",
]
