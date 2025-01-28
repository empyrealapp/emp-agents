from typing import Awaitable, Callable

StrCallable = Callable[..., str | Awaitable[str]]


def add_to_decorated_functions(func: Callable, decorator_name: str):
    """Helper function to add function metadata to the function object."""
    tool_name = func.__name__
    tool_description = func.__doc__ or ""

    if tool_name and tool_description:
        if not hasattr(func, "_tool_metadata"):
            func._tool_metadata = {}

        func._tool_metadata.update(
            {
                "name": tool_name,
                "description": tool_description,
                "decorator": decorator_name,
            }
        )


def tool_method(func: StrCallable):
    """Decorator that marks a method as a protocol tool method.

    This allows the method to be exposed as a tool that can be used by AI agents
    to interact with protocols. The method will be added to the class's tools list
    when the class is created.

    Args:
        func: The method to mark as a protocol tool

    Returns:
        The decorated method
    """
    # Set attribute to mark this as a protocol tool method
    method = func
    setattr(method, "_is_tool_method", True)
    add_to_decorated_functions(method, "tool_method")
    return method


def onchain_action(func: StrCallable):
    """Decorator that marks a method as an onchain action.

    This indicates that the method will submit transactions or otherwise modify blockchain state.
    The method will be added to the class's tools list when the class is created.

    Args:
        func: The method to mark as an onchain action

    Returns:
        The decorated method
    """
    # Set attributes to mark this as a protocol tool method and onchain action
    method = func
    setattr(method, "_is_tool_method", True)
    setattr(method, "_is_onchain_action", True)
    add_to_decorated_functions(method, "onchain_action")
    return method


def view_action(func: StrCallable):
    """Decorator that marks a method as a view-only data fetch action.

    This indicates that the method will only read blockchain state without modifying it.
    The method will be added to the class's tools list when the class is created.

    Args:
        func: The method to mark as a view action

    Returns:
        The decorated method
    """
    # Set attributes to mark this as a protocol tool method and view action
    method = func
    setattr(method, "_is_tool_method", True)
    setattr(method, "_is_view_action", True)
    add_to_decorated_functions(method, "view_action")
    return method


def cachable(func: StrCallable):
    """Decorator that caches the results of a method."""
    cache = {}

    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    # Preserve the original function attributes
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    # Set attribute to mark this as a protocol tool method
    setattr(wrapper, "_is_tool_method", True)
    add_to_decorated_functions(wrapper, "cachable")
    return wrapper
