from typing import Annotated, Literal

import httpx
from eth_rpc import PrivateKeyWallet
from eth_rpc.networks import get_network_by_name
from eth_typeshed import ERC20
from eth_typeshed.multicall import multicall
from typing_extensions import Doc

from emp_agents.implicits import IgnoreDepends
from emp_agents.implicits.manager import ImplicitManager
from emp_agents.models.protocol import SkillSet, onchain_action, view_action


class ERC20Skill(SkillSet):
    """
    Tools for interacting with ERC20 tokens.
    """

    @view_action
    @staticmethod
    async def describe_protocol():
        """Returns the complete protocol specification of the ERC20 protocol"""

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://raw.githubusercontent.com/ethereum/ercs/refs/heads/master/ERCS/erc-20.md"
            )
            return response.text

    @view_action
    @staticmethod
    async def get_token_info(
        network: Annotated[
            Literal["ethereum", "arbitrum", "base"],
            Doc("The network to use.  One of ethereum, arbitrum, or base."),
        ],
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
    ) -> str:
        """Returns the name, symbol and decimals for an ERC20 token"""

        try:
            network_type = get_network_by_name(network)
        except ValueError:
            return "Invalid network"

        token = ERC20[network_type](address=token_address)

        try:
            (name, symbol, decimals) = await multicall[network_type].execute(
                token.name(), token.symbol(), token.decimals()
            )
        except Exception as e:
            return f"Error getting token info: {e}"

        return f"name: {name}; symbol: {symbol}; decimals: {decimals}"

    @view_action
    @staticmethod
    async def get_balance(
        network: Annotated[
            Literal["ethereum", "arbitrum", "base"],
            Doc("The network to use.  One of ethereum, arbitrum, or base."),
        ],
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        address: Annotated[
            str, Doc("The address of the account to get the balance of.")
        ],
        decimals: Annotated[int, Doc("How many decimals the token has")] = 18,
    ) -> str:
        """Returns the balance of an account for an ERC20 token"""

        try:
            network_type = get_network_by_name(network)
        except ValueError:
            return "Invalid network"

        token = ERC20[network_type](address=token_address)
        balance = await token.balance_of(address).get()
        return f"Balance: {balance / 10 ** decimals}"

    @onchain_action
    @staticmethod
    async def transfer(
        network: Annotated[
            Literal["ethereum", "arbitrum", "base"],
            Doc("The network to use.  One of ethereum, arbitrum, or base."),
        ],
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        to_address: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer.")],
        wallet: Annotated[
            PrivateKeyWallet, Doc("The wallet to use to transfer the token.")
        ] = IgnoreDepends(ImplicitManager.lazy_implicit("load_wallet")),
    ) -> str:
        try:
            network_type = get_network_by_name(network)
        except ValueError:
            return "Invalid network"

        token = ERC20[network_type](address=token_address)
        tx = await token.transfer(to_address, amount).execute(wallet)
        return f"Transaction sent: {tx.hash}"

    @onchain_action
    @staticmethod
    async def approve(
        network: Annotated[
            Literal["ethereum", "arbitrum", "base"],
            Doc("The network to use.  One of ethereum, arbitrum, or base."),
        ],
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        private_key: Annotated[
            str, Doc("The private key of the account to transfer from.")
        ],
        spender: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer.")],
    ) -> str:
        try:
            network_type = get_network_by_name(network)
        except ValueError:
            return "Invalid network"

        token = ERC20[network_type](address=token_address)
        wallet = PrivateKeyWallet(private_key=private_key)
        tx = await token.approve(spender, amount).execute(wallet)
        return f"Transaction sent: {tx.hash}"

    @onchain_action
    @staticmethod
    async def transfer_from(
        network: Annotated[
            Literal["ethereum", "arbitrum", "base"],
            Doc("The network to use.  One of ethereum, arbitrum, or base."),
        ],
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        private_key: Annotated[
            str, Doc("The private key of the account to transfer from.")
        ],
        from_address: Annotated[
            str, Doc("The address of the account to transfer from.")
        ],
        to_address: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer.")],
    ) -> str:
        try:
            network_type = get_network_by_name(network)
        except ValueError:
            return "Invalid network"

        token = ERC20[network_type](address=token_address)
        wallet = PrivateKeyWallet(private_key=private_key)
        tx = await token.transfer_from(from_address, to_address, amount).execute(wallet)
        return f"Transaction sent: {tx.hash}"
