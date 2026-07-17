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
from navsim.agents.mimir_rl.mimir_sel_config import MimirConfig

from navsim.agents.mimir_rl.mimir_model_sel import MimirSelModel
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


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)


class MimirSelAgent(AbstractAgent):
    """Agent interface for Mimir selector training."""

    def __init__(
        self,
        config: MimirConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        traj_save_path: str='',
    ):
        """
        Initializes Mimir selector agent.
        :param config: global config of Mimir selector agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self.traj_save_path=traj_save_path
        self._checkpoint_path = checkpoint_path
        self._mimir_model = MimirSelModel(config)
        trainable_prefixes = [
            '_trajectory_head.plan_anchor_scorer_encoder',
            '_trajectory_head.scorer_decoder',
            '_trajectory_head.NC_head',
            '_trajectory_head.DAC_head',
            '_trajectory_head.EP_head',
            '_trajectory_head.C_head',
            '_trajectory_head.TTC_head',
            '_trajectory_head.fine_scorer_decoder',
            '_trajectory_head.fine_NC_head',
            '_trajectory_head.fine_DAC_head',
            '_trajectory_head.fine_EP_head',
            '_trajectory_head.fine_C_head',
            '_trajectory_head.fine_TTC_head',
        ]
        for name, param in self._mimir_model.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in trainable_prefixes)
        self._mimir_model.eval()
        for p in trainable_prefixes:
            m = dict(self._mimir_model.named_modules()).get(p, None)
            if m is not None:
                m.train()
        self.init_from_pretrained()

    def init_from_pretrained(self):
        if self._checkpoint_path:
            checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}

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
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [MimirTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [MimirFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, metric_cache=None,token=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._mimir_model(features,targets=targets,eta=1.0, metric_cache=metric_cache,token=token)

    def compute_trajectory(self, agent_input: AgentInput, token=None) -> Trajectory:
        """Computes the selector-chosen ego trajectory for PDM scoring."""
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            predictions = self.forward(features, token=token)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

            # if self.traj_save_path and token is not None:
            #     traj_save_dir = Path(self.traj_save_path)
            #     traj_save_dir.mkdir(parents=True, exist_ok=True)
            #     np.save(traj_save_dir / f"{token}.npy", poses)

        return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        pred = predictions
        loss = pred.get('loss', None)
        sub_loss_dict = pred.get('sub_loss_dict', None)
        reward_dict = pred.get('reward_dict', None)
        loss_dict = {'loss': loss}
        if sub_loss_dict is not None:
            loss_dict.update(sub_loss_dict)
        if reward_dict is not None:
            loss_dict.update(reward_dict)
        return loss_dict


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
            epochs=20,
            warmup_epochs=1,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [MimirCallback(self._config),
                pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1, monitor="epoch", mode="max"), ]
