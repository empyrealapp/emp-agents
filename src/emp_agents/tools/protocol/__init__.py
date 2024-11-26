from .camelot import CamelotSkill
from .erc20 import ERC20Skill
from .gmx import GmxSkill
from .network import NetworkSkill
from .wallets.simple import SimpleWalletSkill

__all__ = [
    "CamelotSkill",
    "ERC20Skill",
    "NetworkSkill",
    "SimpleWalletSkill",
    "GmxSkill",
]
