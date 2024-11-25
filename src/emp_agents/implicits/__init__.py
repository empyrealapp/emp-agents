from typing import Any, Callable, ParamSpec, TypeVar

from fast_depends import inject

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


__all__ = [
    "ImplicitManager",
    "IgnoreDepends",
    "inject",
]
