from collections.abc import Callable
from typing import Annotated, Optional

import httpx
from eth_rpc import PrivateKeyWallet
from eth_rpc.networks import Network
from eth_typeshed import ERC20
from eth_typeshed.multicall import make_multicall
from typing_extensions import Doc

from emp_agents.implicits import IgnoreDepends, Provider, inject
from emp_agents.models.protocol import SkillSet, onchain_action, view_action

from ..network import NetworkSkill
from ..wallets import SimpleWalletSkill

erc20_scope = Provider()


def load_wallet() -> PrivateKeyWallet | None:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_wallet` to scope this method.
    """
    return SimpleWalletSkill.get_wallet()


def load_network() -> type[Network] | None:
    """
    This can be overridden by using the scope for your agent.
    Use `scope_load_network` to scope this method.
    """
    return NetworkSkill.get_network()


def scope_load_wallet(
    new_load_wallet: Callable[..., PrivateKeyWallet]
) -> tuple[Provider, Callable, Callable]:
    return (erc20_scope, load_wallet, new_load_wallet)


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
    @inject(dependency_overrides_provider=erc20_scope)  # type: ignore[call-overload]
    async def get_token_info(
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        network: Annotated[
            type[Network] | None,
            Doc("The network to use"),
        ] = IgnoreDepends(load_network),
    ) -> str:
        """Returns the name, symbol and decimals for an ERC20 token"""

        if not network:
            return "NOTE: No network set, try setting the network first"

        token = ERC20[network](address=token_address)

        multicall = make_multicall(network)
        try:
            (name, symbol, decimals) = await multicall[network].execute(
                token.name(), token.symbol(), token.decimals()
            )
        except Exception as e:
            return f"Error getting token info: {e}"

        return f"name: {name}; symbol: {symbol}; decimals: {decimals}"

    @view_action
    @staticmethod
    @inject(dependency_overrides_provider=erc20_scope)  # type: ignore[call-overload]
    async def get_balance(
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        address: Annotated[
            str, Doc("The address of the account to get the balance of.")
        ],
        decimals: Annotated[
            Optional[int], Doc("How many decimals the token has")
        ] = None,
        network: Annotated[
            type[Network] | None,
            Doc("The network to use"),
        ] = IgnoreDepends(load_network),
    ) -> str:
        """Returns the balance of an account for an ERC20 token"""
        if network is None:
            return "NOTE: No network set, try setting the network first"

        token = ERC20[network](address=token_address)
        balance = await token.balance_of(address).get()
        if not decimals:
            decimals = await token.decimals().get()
        assert decimals is not None
        return f"Balance: {balance / 10 ** decimals}"

    @onchain_action
    @staticmethod
    @inject(dependency_overrides_provider=erc20_scope)  # type: ignore[call-overload]
    async def transfer(
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        to_address: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer as a decimal.")],
        wallet: Annotated[
            PrivateKeyWallet | None, Doc("The wallet to use to transfer the token.")
        ] = IgnoreDepends(load_wallet),
        network: Annotated[
            type[Network] | None,
            Doc("The network to use"),
        ] = IgnoreDepends(load_network),
    ) -> str:
        """Transfer an amount of an ERC20 token to a recipient"""

        if wallet is None:
            return "NOTE: No wallet set, try setting the wallet first"

        if network is None:
            return "NOTE: No network set, try setting the network first"

        token = ERC20[network](address=token_address)
        decimals = await token.decimals().get()
        token_amount = int(amount * 10**decimals)
        tx = await token.transfer(to_address, token_amount).execute(wallet)
        return f"Transaction sent: {network.block_explorer.url}/tx/{tx.hash}"

    @onchain_action
    @staticmethod
    @inject(dependency_overrides_provider=erc20_scope)  # type: ignore[call-overload]
    async def approve(
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        spender: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer.")],
        network: Annotated[
            type[Network] | None,
            Doc("The network to use"),
        ] = IgnoreDepends(load_network),
        wallet: Annotated[
            PrivateKeyWallet | None, Doc("The wallet to use to transfer the token.")
        ] = IgnoreDepends(load_wallet),
    ) -> str:
        """Approve an amount of an ERC20 token to be spent by a spender"""
        if wallet is None:
            return "NOTE: No wallet set, try setting the wallet first"

        if network is None:
            return "NOTE: No network set, try setting the network first"

        token = ERC20[network](address=token_address)
        tx = await token.approve(spender, amount).execute(wallet)
        return f"Transaction sent: {network.block_explorer.url}/tx/{tx.hash}"

    @onchain_action
    @staticmethod
    @inject(dependency_overrides_provider=erc20_scope)  # type: ignore[call-overload]
    async def transfer_from(
        token_address: Annotated[str, Doc("The address of the ERC20 token.")],
        from_address: Annotated[
            str, Doc("The address of the account to transfer from.")
        ],
        to_address: Annotated[str, Doc("The address of the account to transfer to.")],
        amount: Annotated[float, Doc("The amount to transfer.")],
        network: Annotated[
            type[Network] | None,
            Doc("The network to use"),
        ] = IgnoreDepends(load_network),
        wallet: Annotated[
            PrivateKeyWallet | None, Doc("The wallet to use to transfer the token.")
        ] = IgnoreDepends(load_wallet),
    ) -> str:
        """Transfer an amount of an ERC20 token from an external, approved account to another account"""

        if wallet is None:
            return "NOTE: No wallet set, try setting the wallet first"

        if network is None:
            return "NOTE: No network set, try setting the network first"

        token = ERC20[network](address=token_address)
        tx = await token.transfer_from(from_address, to_address, amount).execute(wallet)
        return f"Transaction sent: {network.block_explorer.url}/tx/{tx.hash}"
