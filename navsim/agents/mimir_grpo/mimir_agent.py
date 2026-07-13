from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent_mimir import AbstractAgent
from navsim.agents.mimir_grpo.mimir_config import MimirConfig

from navsim.agents.mimir_grpo.mimir_model import MimirModel

from navsim.agents.mimir_grpo.mimir_callback import MimirCallback
from navsim.agents.mimir_grpo.mimir_loss import mimir_loss
from navsim.agents.mimir_grpo.mimir_features import MimirFeatureBuilder, MimirTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.mimir_grpo.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class MimirAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: MimirConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        traj_save_path: str='',
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.traj_save_path=traj_save_path
        self._mimir_model = MimirModel(config)
        self.init_from_pretrained()
        if self._config.grpo:
            self._mimir_model.init_grpo_reference()
            self._freeze_non_trajectory_head()

    def _freeze_non_trajectory_head(self) -> None:
        for name, param in self._mimir_model.named_parameters():
            param.requires_grad = (
                name.startswith("_trajectory_head.")
                and not name.startswith("_trajectory_head.old_policy.")
                and name != "_trajectory_head.plan_anchor"
            )
        frozen = sum(param.numel() for param in self._mimir_model.parameters() if not param.requires_grad)
        trainable = sum(param.numel() for param in self._mimir_model.parameters() if param.requires_grad)
        print(f"Mimir GRPO trainable parameters: {trainable}; frozen parameters: {frozen}")

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path is None:
            return
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [MimirTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [MimirFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]=None,
        metric_cache_paths=None,
        token=None,
        goalpoint=None,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._mimir_model(
            features,
            targets=targets,
            goalpoint=goalpoint,
            token=token,
            metric_cache_paths=metric_cache_paths,
            grpo=self.training and torch.is_grad_enabled() and self._config.grpo,
        )
    
    def compute_trajectory(self, agent_input: AgentInput, token=None) -> Tuple[np.ndarray,np.ndarray]:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            # add batch dimension
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            predictions = self.forward(features,token=token)

            poses = predictions['trajectory'].squeeze(0).numpy()# 20 8 3     20 64 8 3
            # poses = predictions['trajectory'].squeeze(0).detach().cpu().numpy()# 20 8 3     20 64 8 3

            # if self.traj_save_path and token is not None:
            #     traj_save_dir = Path(self.traj_save_path)
            #     traj_save_dir.mkdir(parents=True, exist_ok=True)
            #     np.save(traj_save_dir / f"{token}.npy", poses)

            trajectory=Trajectory(poses)

        return trajectory

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if self.training and self._config.grpo:
            return {
                "loss": predictions["loss"],
                "reward": predictions.get("reward"),
                "policy_loss": predictions.get("policy_loss"),
                "bc_loss": predictions.get("bc_loss"),
            }
        return mimir_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._mimir_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if self._config.grpo:
            params = [p for p in self._mimir_model.parameters() if p.requires_grad]
        elif paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._mimir_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._mimir_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if (not self._config.grpo) and paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [MimirCallback(self._config),
        pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1, monitor="epoch", mode="max"), 
        pl.callbacks.ModelCheckpoint(
            monitor="val/trajectory_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch}",
        ),]