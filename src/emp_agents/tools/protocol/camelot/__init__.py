import json
from typing import Annotated
from typing_extensions import Doc

from emp_agents.models.protocol import SkillSet, view_action
from .graph import get_tokens_for_symbol, get_pairs_for_token


class CamelotSkill(SkillSet):
    @view_action
    @staticmethod
    async def get_tokens_for_symbol(
        symbol: Annotated[str, Doc("The symbol to get tokens for")]
    ):
        """
        Get the tokens for a given symbol on Camelot on Arbitrum
        """
        response = await get_tokens_for_symbol(symbol)
        return json.dumps(response)

    @view_action
    @staticmethod
    async def get_pairs_for_token(
        token_address: Annotated[str, Doc("The token address to get pairs for")]
    ):
        """
        Get the pairs for a given token on Camelot on Arbitrum.  This is used to provide the liquidity information for a token.
        """
        response = await get_pairs_for_token(token_address)
        return json.dumps(response)
