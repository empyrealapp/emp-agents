from emp_agents.agents.base import AgentBase
from emp_agents.config.agent_config import PersistentAgentConfig


class PersistentAgent(AgentBase):
    config: PersistentAgentConfig

    @classmethod
    def from_config(cls, config: PersistentAgentConfig) -> "PersistentAgent":
        return cls(
            config=config,
            description=config.description,
            default_model=config.default_model,
            prompt=config.prompt,
            tools=config.tools,
            requires=config.requires,
            personality=config.name,
            **config.extra,
        )

    def perform_action(self):
        print(
            f"Agent {self.config.name} (ID: {self.config.agent_id}) is performing an action."
        )
