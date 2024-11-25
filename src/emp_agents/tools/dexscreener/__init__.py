from typing import Annotated, Literal

from typing_extensions import Doc

from emp_agents.models.protocol import SkillSet, tool_method
from emp_agents.tools.dexscreener.api import DexScreenerApi


class DexScreenerSkill(SkillSet):
    """
    Skill for interacting with DexScreener API
    Documentation: https://docs.dexscreener.com/api/reference
    """

    @tool_method
    @staticmethod
    async def search_pairs(
        query: Annotated[str, Doc("The search query to find trading pairs")],
    ) -> str:
        """Search for pairs matching query (rate-limit 300 requests per minute)"""
        return await DexScreenerApi.search_pairs(query)

    @tool_method
    @staticmethod
    async def get_pair_by_chain(
        chain_id: Annotated[
            Literal["ethereum", "solana", "arbitrum", "base", "bsc"],
            Doc("The chain to search on"),
        ],
        pair_id: Annotated[str, Doc("The pair contract address")],
    ) -> str:
        """Retrieve trading pairs by specified chain and pair address (rate-limit 300 requests per minute)"""
        return await DexScreenerApi.get_pair_by_chain(chain_id, pair_id)

    @tool_method
    @staticmethod
    async def find_pairs_by_tokens(
        token_addresses: Annotated[list[str], Doc("List of token addresses (max 30)")]
    ) -> str:
        """Get pairs by token addresses (rate-limit 300 requests per minute)"""
        return await DexScreenerApi.find_pars_by_tokens(token_addresses)

    @tool_method
    @staticmethod
    async def get_token_profiles() -> str:
        """Get the latest token profiles (rate-limit 60 requests per minute)"""
        return await DexScreenerApi.get_token_profiles()

    @tool_method
    @staticmethod
    async def get_latest_boosted_tokens() -> str:
        """Get the latest boosted tokens (rate-limit 60 requests per minute)"""
        return await DexScreenerApi.get_latest_boosted_tokens()

    @tool_method
    @staticmethod
    async def get_top_boosted_tokens() -> str:
        """Get the tokens with most active boosts (rate-limit 60 requests per minute)"""
        return await DexScreenerApi.get_top_boosted_tokens()
