from fast_depends.dependencies.model import Depends


class IgnoreDepends(Depends):
    """
    A Depends that ignores the implicit argument for the tools calls.
    This means the implicit argument must be set before the tools call,
    otherwise the function will fail.
    """
