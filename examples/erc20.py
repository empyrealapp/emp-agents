import asyncio
import os

from eth_rpc import set_alchemy_key

from emp_agents.agents.skills import SkillsAgent
from emp_agents.logger import make_verbose
from emp_agents.tools.protocol.erc20 import ERC20Skill
from emp_agents.tools.protocol.network import NetworkSkill
from emp_agents.tools.protocol.wallets import SimpleWalletSkill
from emp_agents.types import OpenAIModelType

make_verbose(False)

if alchemy_key := os.environ.get("ALCHEMY_KEY"):
    set_alchemy_key(alchemy_key)


class ERC20Agent(SkillsAgent):
    def _load_implicits(self):
        if private_key := os.environ.get("EMP_AGENT_PRIVATE_KEY"):
            SimpleWalletSkill.set_private_key(private_key)
        if network := os.environ.get("EMP_AGENT_NETWORK"):
            NetworkSkill.set_network(network)


agent = ERC20Agent(
    skills=[
        ERC20Skill,
        NetworkSkill,
        SimpleWalletSkill,
    ],
    default_model=OpenAIModelType.gpt4o_mini,
)


if __name__ == "__main__":
    asyncio.run(agent.run())
