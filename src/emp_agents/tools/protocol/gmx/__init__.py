from typing import Annotated, Literal

import httpx
from typing_extensions import Doc

from emp_agents.logger import logger
from emp_agents.models.protocol import SkillSet, view_action


class GmxSkill(SkillSet):
    @view_action
    @staticmethod
    async def get_tokens_address_dict(
        chain: Annotated[
            Literal["arbitrum", "avalanche"],
            Doc("The chain to query for tokens"),
        ] = "arbitrum",
    ):
        """
        Query the GMX infra api for to generate dictionary of tokens available on v2.  Returns
        a JSON of tokens available on the given chain.
        """

        url = {
            "arbitrum": "https://arbitrum-api.gmxinfra.io/tokens",
            "avalanche": "https://avalanche-api.gmxinfra.io/tokens",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url[chain])

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the JSON response
                token_infos = response.json()["tokens"]
            else:
                logger.error(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")

        token_address_dict = {}

        for token_info in token_infos:
            token_address_dict[token_info["address"]] = token_info

        return token_address_dict
