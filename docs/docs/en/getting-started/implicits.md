Implicits are used to inject dependencies into skills.  This is done using the `@inject` decorator, and the `Depends` and `IgnoreDepends` classes.

## Depends

The `IgnoreDepends` class is used to provide a dependency that will be injected, but can not be provided by the agent.  This is important because agents are not aware of types outside of text, so it allows you to use context created by the agent to be transformed and injected optionally to the argument.  `Depends` is used to inject a dependency that can be provided by the agent,
but is optional and can be injected if it is currently scoped.

## Provider

You can also define a skill that uses `Provider` to scope dependencies.  This allows you to override the dependency for a specific skill, so you can provide a default handler, but also override this to use an external data source, or some other format for the skill.  This is similar to creating an interface with a default implementation, but users can easily create their own management of this persistent state.


## Example

```python
import asyncio
from typing import Optional, Callable

from emp_agents.models.protocol import SkillSet, tool_method, view_action
from emp_agents.implicits import Provider, Depends, IgnoreDepends, inject
from emp_agents.agents import SkillsAgent

from contextvars import ContextVar


# we define two context variables that will be used to provide the numerator and denominator
_numerator: ContextVar[Optional[str]] = ContextVar("_numerator", default=None)
_denominator: ContextVar[Optional[int]] = ContextVar("_denominator", default=None)


# we define a scope so the provider can be overridden
math_scope = Provider()


# this is the default loader for numerator, which uses the context variable
# it uses a string because it can be provided by the agent
def load_numerator() -> str:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_numerator` to scope this method.
    """
    return _numerator.get()


# this is the default loader for denominator, which uses the context variable and provides an integer
# this can not be provided by the agent
def load_denominator() -> int:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_denominator` to scope this method.
    """
    return _denominator.get()


# this is a helper function to override the load_numerator function
def scope_load_numerator(
    new_load_numerator: Callable[..., int]
) -> tuple[Provider, Callable, Callable]:
    return (math_scope, load_numerator, new_load_numerator)


# this is a helper function to override the load_denominator function
def scope_load_denominator(
    new_load_denominator: Callable[..., int]
) -> tuple[Provider, Callable, Callable]:
    return (math_scope, load_denominator, new_load_denominator)


# lets define a skill that uses the numerator and denominator
class FractionSkill(SkillSet):
    """
    Tools for interacting with fractions.
    """

    @view_action
    @staticmethod
    @inject(dependency_overrides_provider=math_scope)
    async def make_fraction(
        x: str = Depends(load_numerator),
        divisor: int = IgnoreDepends(load_denominator),
    ) -> str:
        """divide two values, and provide the fraction as a string"""
        return str(int(x) / divisor)

    @tool_method
    @staticmethod
    def update_denominator(new_denominator: str):
        if not new_denominator.isdigit():
            return "thats a bad value"
        _denominator.set(int(new_denominator))
        return "denominator updated"


class FractionAgent(SkillsAgent):
    def _load_implicits(self):
        _numerator.set("42")
        _denominator.set(2)


agent = FractionAgent(
    personality="be brief.",
    skills=[
        FractionSkill,
    ],
)

print(asyncio.run(agent.answer("Make a fraction")))
# Output: The fraction is \( \frac{21}{1} \).

# update the denominator
print(asyncio.run(agent.answer("Update the denominator to 10")))

# make a fraction
print(asyncio.run(agent.answer("Make a fraction")))
# Output: The fraction created is \( \frac{1}{10} \) which is equivalent to 0.1. If you need a different numerator, let me know!


# Lets create a new agent that overrides the numerator and denominator functions
agent = FractionAgent(
    skills=[FractionSkill],
    personality="be brief.",
    scopes=[
        scope_load_numerator(lambda: "1000"),
        scope_load_denominator(lambda: 10),
    ],
)

print(asyncio.run(agent.answer("Make a fraction")))
# Output: The fraction is \( \frac{100}{1} \).
```
