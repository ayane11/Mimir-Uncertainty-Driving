from dataclasses import dataclass
from typing import Tuple

from navsim.agents.mimir_rl.mimir_config import MimirConfig as BaseMimirConfig


@dataclass
class MimirConfig(BaseMimirConfig):
    """Mimir config with selector after-training knobs."""

    num_groups: int = 8
    num_samples: int = 8
    fusion_base_weight: float = 1.0
    fusion_wm_weight: float = 0.5
    weight: float = 0.2
    fusion_coarse_weights: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0)
