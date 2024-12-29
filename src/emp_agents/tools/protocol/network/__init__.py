from typing import Annotated, Literal, Optional, cast

from eth_rpc.networks import Network, get_network_by_name
from typing_extensions import Doc

from emp_agents.models.protocol import SkillSet, tool_method

_network: Optional[type[Network]] = None


NetworkOptions = Literal[
    "ethereum",
    "sepolia",
    "base",
    "BaseSepolia",
    "arbitrum",
    "ArbitrumSepolia",
]


class NetworkSkill(SkillSet):
    """
    Tools for controlling the current network
    """

    @tool_method
    @staticmethod
    def set_network(
        network: Annotated[
            NetworkOptions,
            Doc("The network to set"),
        ],
    ):
        """
        Set the current network
        """
        global _network
        try:
            _network = get_network_by_name(network)
        except ValueError:
            return f"Invalid network: {network}"
        return f"network set to {network}"

    @tool_method
    @staticmethod
    def get_network() -> str:
        """
        Get the current network
        """
        if not _network:
            return "No network set, try setting the network first"
        return f"current network: {_network.name}"

    @tool_method
    @staticmethod
    def make_block_explorer_link(tx_hash: str) -> str:
        if not _network:
            return "No network set, try setting the network first"
        return f"{_network.block_explorer.url}/tx/{tx_hash}"

    @staticmethod
    def get_network_type() -> type[Network] | None:
        return _network

    @staticmethod
    def get_network_str() -> NetworkOptions | None:
        if not _network:
            return None
        return cast(NetworkOptions, _network.name)
