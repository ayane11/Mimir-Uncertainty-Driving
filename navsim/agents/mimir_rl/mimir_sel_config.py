from dataclasses import dataclass

from navsim.agents.mimir_rl.mimir_config import MimirConfig as BaseMimirConfig


@dataclass
class MimirConfig(BaseMimirConfig):
    """Mimir config with selector after-training knobs."""

    num_groups: int = 8
