from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
import copy
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.mimir_rl.mimir_rl_config import MimirConfig

from navsim.agents.mimir_rl.mimir_model_rl import MimirRlModel

from navsim.agents.mimir_rl.mimir_callback import MimirCallback
from navsim.agents.mimir_rl.mimir_loss import mimir_loss
from navsim.agents.mimir_rl.mimir_features import MimirFeatureBuilder, MimirTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.mimir_rl.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import torch.nn.functional as F
import numpy as np
import re

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


class MimirRlAgent(AbstractAgent):
    """Agent interface for Mimir RL training."""

    def __init__(
        self,
        config: MimirConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes Mimir RL agent.
        :param config: global config of Mimir agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._mimir_model = MimirRlModel(config)
        for name, param in self._mimir_model.named_parameters():
            if not name.startswith("_trajectory_head"):
                param.requires_grad = False
        self._apply_module_freeze()
        self.init_from_pretrained()

    def _apply_module_freeze(self) -> None:
        """Keep frozen modules in eval mode while the trajectory head trains."""
        for name, module in self._mimir_model.named_modules():
            if name and not name.startswith("_trajectory_head"):
                module.eval()
        self._mimir_model._trajectory_head.train()

    def train(self, mode: bool = True):
        """Set training mode while preserving the frozen-module split.

        Lightning recursively calls ``train()`` on the agent every epoch.  The
        recursive call would otherwise re-enable frozen BatchNorm buffers and
        Dropout modules after the initialization-time ``eval()`` calls.
        """
        super().train(mode)
        if mode:
            self._apply_module_freeze()
        return self

    def init_from_pretrained(self):
        if self._checkpoint_path:
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

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, metric_cache=None, token=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._mimir_model(features,targets=targets, eta=1.0, metric_cache=metric_cache, token=token)

    def compute_trajectory(self, agent_input: AgentInput, token=None) -> Trajectory:
        """Computes the ego vehicle trajectory for PDM scoring."""
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            predictions = self.forward(features, token=token)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        loss = predictions['loss']
        reward = predictions['reward']
        sub_rewards = predictions.get('sub_rewards', None)
        loss_dict = {'loss': loss, 'reward':reward}
        for key in (
            'reward_mean',
            'reward_max',
            'reward_std',
            'reward_gt_mean',
            'positive_rate',
            'adv_positive_rate',
            'rl_loss',
            'il_loss',
        ):
            value = predictions.get(key)
            if value is not None:
                loss_dict[key] = value
        if sub_rewards is not None:
            loss_dict.update(sub_rewards) # add sub rewards to loss_dict if available
        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._mimir_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
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
        
        if paramwise_cfg:
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
        if paramwise_cfg:
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
            epochs=10,
            warmup_epochs=1,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [MimirCallback(self._config),
                pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1, monitor="epoch", mode="max"), ]
