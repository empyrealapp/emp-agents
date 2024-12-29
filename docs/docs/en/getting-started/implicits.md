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

# we define two context variables for the skill state
_numerator: ContextVar[Optional[str]] = ContextVar("_numerator", default=None)
_denominator: ContextVar[Optional[int]] = ContextVar("_denominator", default=None)

# a scope is used to wrap the scope state management
math_scope = Provider()


# This is the default loader for numerator, which uses the context variable.
# It uses a string because it can also be provided by the agent
def load_numerator() -> str | None:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_numerator` to scope this method.
    """
    return _numerator.get()


# This is the default loader for denominator, which uses the context variable and provides an integer
# this can not be provided by the agent
def load_denominator() -> int | None:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_denominator` to scope this method.
    """
    return _denominator.get()


# Helper function to override the load_numerator function
def scope_load_numerator(
    new_load_numerator: Callable[..., int]
) -> tuple[Provider, Callable, Callable]:
    return (math_scope, load_numerator, new_load_numerator)


# Helper function to override the load_denominator function
def scope_load_denominator(
    new_load_denominator: Callable[..., int]
) -> tuple[Provider, Callable, Callable]:
    return (math_scope, load_denominator, new_load_denominator)


# Define a skill class that utilizes the numerator and denominator values
class FractionSkill(SkillSet):
    """
    Tools for interacting with fractions.
    """

    @view_action
    @inject(dependency_overrides_provider=math_scope)
    async def make_fraction(
        x: int = Depends(load_numerator),
        divisor: int = IgnoreDepends(load_denominator),
    ) -> str:
        """divide two values, and provide the fraction as a string"""
        return str(x / divisor)

    @tool_method
    def update_denominator(new_denominator: str):
        if not new_denominator.isdigit():
            return "Thats a bad value"
        _denominator.set(int(new_denominator))
        return "denominator updated"


class FractionAgent(SkillsAgent):
    def _load_implicits(self):
        _numerator.set("42")
        _denominator.set(2)


# We create an async function, which will keep the context shared.
# Since each thread has its own context, this is a good reason to use
# a different mechanism for persistence like a database as a scoped override.
# See: https://docs.python.org/3/library/contextvars.html#contextvars.Context
async def main():
    agent = FractionAgent(
        prompt="You are a helpful assistant that provides terse responses as fractions",
        skills=[
            FractionSkill,
        ],
    )

    print(await agent.answer("Make a fraction"))
    # Output: The fraction is \( \frac{21}{1} \).

    # Update the denominator
    await agent.answer("Update the denominator to 10")

    # Make a fraction
    print(await agent.answer("Make a fraction"))
    # Output: he fraction is \( \frac{4.2}{1} \) with a denominator of 10.

    # Lets create a new agent that overrides the numerator and denominator functions
    agent2 = FractionAgent(
        skills=[FractionSkill],
        prompt="You are a helpful assistant that provides terse responses as fractions",
        scopes=[
            scope_load_numerator(lambda: "1000"),
            scope_load_denominator(lambda: 10),
        ],
    )

    print(await agent2.answer("Make a fraction"))
    # Output: The fraction is \( \frac{100}{1} \).


if __name__ == "__main__":
    asyncio.run(main())
```
