A key capability of agent systems is their ability to interact with external data sources and services through tool integrations. This framework provides a flexible and robust architecture for managing these integrations. Tools can be dynamically added to or removed from agents as needed, and related tools can be logically grouped into reusable `Skill` objects to promote modularity and maintainability.

Imagine you have two functions you want the agent to be able to call:

```python
import httpx
import requests


def get_lyrics(artist: str, song: str) -> str:
    url = f"https://api.lyrics.ovh/v1/{artist}/{song}"
    response = requests.get(url)
    return response.json()["lyrics"]


async def get_cat_fact() -> str:
    url = "https://catfact.ninja/fact"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.json()["fact"]
```

!!! tip
    Notice it supports both sync and async functions.

Now we can add clear documentation to thse functions to enable our agent to understand how to use it.  The arguments must be annotated with the Doc object, and the function must have a docstring to annotate its intended use.  A tool should always return a string, as that is what the LLM is expecting from all prompts:

```python
from typing import Annotated
from typing_extensions import Doc

import requests
import httpx

def get_lyrics(
    artist: Annotated[str, Doc("The name of the artist")],
    song: Annotated[str, Doc("The name of the song")]
) -> str:
    """
    Get the lyrics for a song by an artist
    """
    url = f"https://api.lyrics.ovh/v1/{artist}/{song}"
    response = requests.get(url)
    return response.json()["lyrics"]


async def get_cat_fact() -> str:
    """
    Get a random cat fact
    """
    url = "https://catfact.ninja/fact"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.json()["fact"]
```

We can now construct an agent that has access to these tools. The agent will be able to understand the tools' functionality through their documentation and type hints, and can use them appropriately in conversations:

```python
from emp_agents import AgentBase

prompt = """You are a helpful assistant that can provide song lyrics and cat facts.
You are very serious, and keep your responses brief and professional.
"""

agent = AgentBase(
    prompt=prompt,
    tools=[get_lyrics, get_cat_fact]
)
```
