from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.mimir_rl.mimir_rl_config import MimirConfig
from navsim.agents.mimir_rl.mimir_backbone import MimirBackbone
from navsim.agents.mimir_rl.mimir_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from navsim.agents.mimir_rl.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.mimir_rl.modules.blocks import (
    linear_relu_ln,
    bias_init_with_prob,
    gen_sineembed_for_position,
    GridSampleCrossBEVAttention,
    GridSampleCrossBEVAttention_navi,
    GridSampleCrossBEVAttention_naviscore,
)
from navsim.agents.mimir_rl.modules.multimodal_loss import LossComputer
from typing import Any, List, Dict, Optional, Union, Tuple
import math
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.evaluate.pdm_score import pdm_score, pdm_score_para
import itertools, os
import lzma
import pickle
import concurrent.futures as cf
import threading
import multiprocessing as mp
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex as WIdx,
)
import matplotlib as mpl
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

def _transform_navi_to_camera_tensor(
    navi: torch.Tensor,
    sensor2lidar_rotation: torch.Tensor,
    sensor2lidar_translation: torch.Tensor,
) -> torch.Tensor:
    lidar2cam_r = sensor2lidar_rotation.inverse()
    lidar2cam_t = -torch.matmul(lidar2cam_r, sensor2lidar_translation)

    locs_homo = torch.cat([navi, torch.ones_like(navi[:, :1])], dim=-1)
    lidar2cam_rt = torch.eye(4, device=navi.device).to(navi)
    lidar2cam_rt[:3, :3] = lidar2cam_r
    lidar2cam_rt[:3, 3] = lidar2cam_t
    return torch.matmul(lidar2cam_rt, locs_homo.T).T[:, :3]


def _project_points_to_image_tensor(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    pc_homo = torch.cat(
        [points, torch.ones((num_points, 1), device=points.device, dtype=points.dtype)],
        dim=-1,
    )
    intrinsics_pad = torch.eye(4, device=points.device, dtype=points.dtype)
    intrinsics_pad[:3, :3] = intrinsics

    proj = torch.matmul(pc_homo, intrinsics_pad.T)
    z = proj[:, 2:3].clamp(min=eps)
    xy = proj[:, 0:2] / z
    in_fov = proj[:, 2] > eps

    if image_shape is not None:
        height, width = image_shape
        u, v = xy[:, 0], xy[:, 1]
        in_bounds = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        in_fov = in_fov & in_bounds

    return xy, in_fov


def extract_feature_values_at_navi_batched(
    feature_map: torch.Tensor,
    navi_tensor: torch.Tensor,
    sensor2lidar_rotation: torch.Tensor,
    sensor2lidar_translation: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    batch_size, channels, height_feat, width_feat = feature_map.shape
    _, num_points, _ = navi_tensor.shape
    height_img, width_img = image_shape
    feature_values = torch.zeros((batch_size, channels, num_points), device=navi_tensor.device)

    for batch_idx in range(batch_size):
        navi_cam = _transform_navi_to_camera_tensor(
            navi_tensor[batch_idx],
            sensor2lidar_rotation[batch_idx],
            sensor2lidar_translation[batch_idx],
        )
        pixel_coords, valid_mask = _project_points_to_image_tensor(
            navi_cam,
            intrinsics[batch_idx],
            image_shape=image_shape,
        )
        pixel_coords_scaled = pixel_coords.clone()
        pixel_coords_scaled[:, 0] *= width_feat / width_img
        pixel_coords_scaled[:, 1] *= height_feat / height_img

        u = pixel_coords_scaled[:, 0].round().long().clamp(0, width_feat - 1)
        v = pixel_coords_scaled[:, 1].round().long().clamp(0, height_feat - 1)
        feature_values[batch_idx] = feature_map[batch_idx, :, v, u] * valid_mask.unsqueeze(0)

    return feature_values


def load_navis_from_np(data_dict, token, num_goal_points: int = 1):
    tokens = [token] if isinstance(token, str) else list(token)
    data = []
    for t in tokens:
        navis = np.asarray(data_dict[t], dtype=np.float32)
        if navis.ndim == 1:
            navis = navis[None, :]
        data.append(navis[:num_goal_points, :2])
    return torch.from_numpy(np.stack(data, axis=0))


class RewardConvNet(nn.Module):
    def __init__(self, input_channels: int, conv1_out_channels: int, conv2_out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=conv1_out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return self.pool(x)


def _pairwise_subscores(scorer):
    """
    从已调用过 score_proposals 的 PDMScorer 中
    拆出 7 个子指标和最终分数，全部 shape=(G,)，顺序与 proposal id 对齐
    返回 dict[str, np.ndarray]
    """
    mm   = scorer._multi_metrics                # (3, N)
    wm   = scorer._weighted_metrics.copy()      # <<< 一定要 copy !
    prod = mm.prod(axis=0)                      # (N,)

    wcoef  = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw             # (N,)

    # ---------- progress 归一化（与 _pairwise_scores 完全一致） ----------
    raw_prog    = prog_raw * prod
    raw_prog_gt = raw_prog[0]
    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    # ---------- 加权指标 ----------
    wscore = (wm * wcoef[:, None]).sum(axis=0) / wcoef.sum()

    return {
        "no_collision"  : mm[MultiMetricIndex.NO_COLLISION,        1:].copy(),
        "drivable_area" : mm[MultiMetricIndex.DRIVABLE_AREA,       1:].copy(), # 乘法版
        "progress"      : wm[WeightedMetricIndex.PROGRESS,         1:].copy(),
        "ttc"           : wm[WeightedMetricIndex.TTC,              1:].copy(),
        "comfort"       : wm[WeightedMetricIndex.COMFORTABLE,      1:].copy(),
        "dir_weighted"  : wm[WeightedMetricIndex.DRIVING_DIRECTION,1:].copy(),
        "final"         : prod[1:] * wscore[1:],                               # 总分
    }
def _pairwise_scores(scorer) -> np.ndarray:
    """
    使用 scorer 在 batch 模式下缓存的中间结果，
    重新计算“GT (索引0) vs 每条候选”的得分。
    返回 shape = (N-1,)  float32。
    """
    # --- 取中间量 ---------------------------------------------------
    mm   = scorer._multi_metrics            # (M_mul, N)
    wm   = scorer._weighted_metrics.copy()  # (M_wgt, N)  (复制以便我们改进程)
    prog_raw = scorer._progress_raw         # (N,)
    weight_coef = scorer._config.weighted_metrics_array  # (M_wgt,)

    N = mm.shape[1]                         # proposals = 1(GT) + G
    assert N >= 2, "Need at least GT + 1 proposal"

    # --- 计算乘法指标乘积 ------------------------------------------
    multi_prod = mm.prod(axis=0)            # (N,)

    # --- 重新归一化 progress，每条候选只与 GT 对标 ------------------
    raw_prog    = prog_raw * multi_prod     # (N,)
    raw_prog_gt = raw_prog[0]

    max_pair    = np.maximum(raw_prog_gt, raw_prog[1:])           # (G,)
    thresh      = scorer._config.progress_distance_threshold

    # 若 max_pair > thresh → 按比例归一；否则看 collision 情况
    norm_prog   = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)                                         # (G,)

    # 把 progress 行（WeightedMetricIndex.PROGRESS）替换成新的
    wm[WIdx.PROGRESS, 1:] = norm_prog

    # --- 计算 weighted_metric_scores（与 _aggregate_scores 同式） ----
    weighted_scores = (wm[:, 1:] * weight_coef[:, None]).sum(axis=0)
    weighted_scores /= weight_coef.sum()                         # (G,)

    # --- 最终得分 = 乘法指标 × 加权指标 -----------------------------
    final_scores = multi_prod[1:] * weighted_scores              # (G,)

    return final_scores.astype(np.float32)                       # (G,)

def _pdm_worker(args):
    cache, traj_np = args
    # if isinstance(cache, str): 
    with lzma.open(cache, "rb") as f:
        metric_cache = pickle.load(f)
    # else:
    #     metric_cache = cache
    results = pdm_score_para(
        metric_cache=metric_cache,
        model_trajectory=traj_np,                # (G, T, C)
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,                    # 全局对象，见 initializer
        scorer=SCORER,
    )
    scores = _pairwise_scores(SCORER)
    subscores  = _pairwise_subscores(SCORER)
    return scores.astype(np.float32), metric_cache, subscores  # (G,)

def _init_pool(sim_cfg, scorer_cfg):
    global SIMULATOR, SCORER
    SIMULATOR = instantiate(sim_cfg)
    SCORER    = instantiate(scorer_cfg)

    
class MimirRlModel(nn.Module):
    """Torch module for Mimir RL training."""

    def __init__(self, config: MimirConfig):
        """
        Initializes Mimir RL torch module.
        :param config: global config dataclass of Mimir.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = MimirBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        if self._config.status_norm:
            self._status_encoding = nn.Linear(4 + 1 + 1, config.tf_d_model)
        else:
            self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        self.weight_score = None
        if self._config.unc_path:
            self.weight_score = np.load(self._config.unc_path, allow_pickle=True).item()

        self.goalpoints = None
        if self._config.navi_path:
            self.goalpoints = np.load(self._config.navi_path, allow_pickle=True).item()

        if self._config.use_wm:
            self._wm_num_future_frames = int(getattr(config, "wm_num_future_frames", 3))
            self._wm_window_num_poses = config.trajectory_sampling.num_poses - self._wm_num_future_frames + 1
            if self._wm_window_num_poses <= 0:
                raise ValueError(
                    "WM sliding-window trajectory length must be positive. "
                    f"Received num_poses={config.trajectory_sampling.num_poses} and "
                    f"wm_num_future_frames={self._wm_num_future_frames}."
                )
            self._wm_reward_num_future_frames = int(getattr(config, "wm_reward_num_future_frames", 1))
            self._wm_reward_window_num_poses = config.trajectory_sampling.num_poses - self._wm_reward_num_future_frames + 1
            if self._wm_reward_window_num_poses <= 0:
                raise ValueError(
                    "WM reward sliding-window trajectory length must be positive. "
                    f"Received num_poses={config.trajectory_sampling.num_poses} and "
                    f"wm_reward_num_future_frames={self._wm_reward_num_future_frames}."
                )
            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.wm_num_head,
                dim_feedforward=config.wm_d_ffn,
                dropout=config.wm_dropout,
                batch_first=True,
            )
            self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, config.wm_num_layers)
            wm_action_dim = self._wm_window_num_poses * 2
            self._wm_action_aware_encoder = nn.Sequential(
                nn.Linear(config.tf_d_model + wm_action_dim, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_model, config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_model, config.tf_d_model),
            )
            if self._wm_reward_window_num_poses == self._wm_window_num_poses:
                self._wm_reward_action_aware_encoder = self._wm_action_aware_encoder
            else:
                wm_reward_action_dim = self._wm_reward_window_num_poses * 2
                self._wm_reward_action_aware_encoder = nn.Sequential(
                    nn.Linear(config.tf_d_model + wm_reward_action_dim, config.tf_d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.tf_d_model, config.tf_d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.tf_d_model, config.tf_d_model),
                )
            self._wm_bev_feature_head = nn.Sequential(
                nn.Conv2d(config.tf_d_model, config.bev_features_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(config.bev_features_channels, config.bev_features_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            wm_reward_steps = self._wm_reward_num_future_frames + 1
            self.reward_conv_net = RewardConvNet(
                input_channels=config.tf_d_model * wm_reward_steps,
                conv1_out_channels=config.tf_d_model,
                conv2_out_channels=config.tf_d_model,
            )
            self.reward_cat_head = nn.Sequential(
                nn.Linear(config.tf_d_model * (wm_reward_steps + 1), config.tf_d_model),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_model, config.tf_d_model),
            )
            self._wm_reward_head = nn.Sequential(
                nn.Linear(config.tf_d_model, config.tf_d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_model // 2, 1),
            )


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None, eta=0.0, metric_cache=None, cal_pdm=True,token=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        if metric_cache is None:
            metric_cache = features.get("metric_cache_path")
        if token is None:
            token = features.get("token")
        if self._config.status_norm and self._config.training == False:
            vle = torch.norm(status_feature[:, 4:6], dim=-1, keepdim=True)
            acc = torch.norm(status_feature[:, 6:8], dim=-1, keepdim=True)
            dot_product = torch.sum(status_feature[:, 4:6] * status_feature[:, 6:8], dim=-1, keepdim=True)
            tag_vle = torch.where(status_feature[:, 4:5] >= 0, 1.0, -1.0)
            tag_acc = torch.where(dot_product > 0, 1.0, -1.0)
            status_feature = torch.cat([status_feature[:, :4], tag_vle * vle, tag_acc * acc], dim=-1)

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, img_feature = self._backbone(camera_feature, lidar_feature)
        if self._config.training and self._config.use_proj_image:
            navis = features["gt_trajs"][:, -1:, :]
            navis[:, :, 2] = 0.0
            rotation = features["sensor2lidar_rot"][:, 0].to(navis)
            translation = features["sensor2lidar_trans"][:, 0].to(navis)
            intrinsic = features["intrinsic"][:, 0].to(navis)
            pix = extract_feature_values_at_navi_batched(
                img_feature[:, :, :, 17:-17],
                navis,
                rotation,
                translation,
                intrinsic,
                (1024, 1920),
            )
        else:
            pix = None

        if self.weight_score:
            points_score = load_navis_from_np(self.weight_score, token, self._config.num_goal_points).to(bev_feature)
        else:
            points_score = None

        if self.goalpoints:
            goalpoint = load_navis_from_np(self.goalpoints, token, self._config.num_goal_points).to(bev_feature)
        else:
            goalpoint = None

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        with torch.no_grad():
            old_pred = self._trajectory_head(trajectory_query,agents_query,cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=pix,eta=eta, old_pred=None,metric_cache=metric_cache,cal_pdm=cal_pdm,token=token,goalpoint=goalpoint,points_score=points_score)
        pred = self._trajectory_head(trajectory_query,agents_query,cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=pix,eta=eta,old_pred=old_pred,metric_cache=metric_cache,cal_pdm=cal_pdm,goalpoint=goalpoint,points_score=points_score)
        if 'reward' not in pred:
            pred['reward'] = old_pred['reward']
        if 'sub_rewards' not in pred:
            pred['sub_rewards'] = old_pred['sub_rewards']
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        if self._config.use_wm:
            self._add_wm_candidate_selection_outputs(output, keyval)
            self._apply_wm_candidate_selection(output)

            if targets is not None and "trajectory" in output:
                wm_training_trajectory = output["trajectory"]
                if wm_training_trajectory.ndim == 4:
                    wm_training_trajectory = wm_training_trajectory[:, -1]
                output["wm_future_bev_semantic_map"] = self._rollout_wm_future_bev_semantic_maps(
                    initial_latent=keyval,
                    trajectory=wm_training_trajectory.to(keyval),
                )

        return output

    def _rollout_wm_future_bev_semantic_maps(
        self,
        initial_latent: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        wm_future_latents = self._rollout_wm_future_latents(
            initial_latent=initial_latent,
            trajectory=trajectory,
            num_future_frames=self._wm_num_future_frames,
            window_num_poses=self._wm_window_num_poses,
            action_encoder=self._wm_action_aware_encoder,
        )
        return self._decode_wm_future_bev_semantic_maps(wm_future_latents)

    def _rollout_wm_future_latents(
        self,
        initial_latent: torch.Tensor,
        trajectory: torch.Tensor,
        num_future_frames: int,
        window_num_poses: int,
        action_encoder: nn.Module,
    ) -> torch.Tensor:
        wm_latent = initial_latent
        wm_future_latents = []

        for future_offset in range(1, num_future_frames + 1):
            wm_trajectory = self._get_wm_trajectory_window(
                trajectory=trajectory,
                window_start=future_offset - 1,
                window_num_poses=window_num_poses,
            )
            wm_latent = self._predict_next_latent(
                prev_latent=wm_latent,
                prev_trajectory=wm_trajectory,
                action_encoder=action_encoder,
            )
            wm_future_latents.append(wm_latent)

        return torch.stack(wm_future_latents, dim=1)

    def _decode_wm_future_bev_semantic_maps(self, wm_future_latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_future_frames, num_tokens, channels = wm_future_latents.shape
        flat_latents = wm_future_latents.reshape(batch_size * num_future_frames, num_tokens, channels)
        flat_maps = self._decode_wm_bev_semantic_map(flat_latents)
        return flat_maps.view(
            batch_size,
            num_future_frames,
            flat_maps.shape[1],
            flat_maps.shape[2],
            flat_maps.shape[3],
        )

    def _add_wm_candidate_selection_outputs(
        self,
        output: Dict[str, torch.Tensor],
        keyval: torch.Tensor,
    ) -> None:
        candidate_trajectories = output.get("candidate_trajectories")
        candidate_logits = output.get("candidate_logits")
        if candidate_trajectories is None or candidate_logits is None:
            return

        batch_size, num_candidates = candidate_logits.shape
        if num_candidates <= 0:
            return

        candidate_indices = torch.arange(
            num_candidates,
            device=candidate_logits.device,
        )[None].expand(batch_size, -1)
        max_wm_candidates = self._get_wm_candidate_selection_count(output, num_candidates)
        if max_wm_candidates < num_candidates:
            candidate_topk_indices = candidate_logits.topk(max_wm_candidates, dim=1).indices
            trajectory_gather_index = candidate_topk_indices[:, :, None, None].expand(
                -1,
                -1,
                candidate_trajectories.shape[2],
                candidate_trajectories.shape[3],
            )
            candidate_trajectories = torch.gather(
                candidate_trajectories,
                dim=1,
                index=trajectory_gather_index,
            )
            candidate_logits = torch.gather(candidate_logits, dim=1, index=candidate_topk_indices)
            candidate_indices = torch.gather(candidate_indices, dim=1, index=candidate_topk_indices)
            num_candidates = max_wm_candidates

        flat_keyval = keyval[:, None].expand(-1, num_candidates, -1, -1).reshape(
            batch_size * num_candidates,
            keyval.shape[1],
            keyval.shape[2],
        )
        flat_trajectories = candidate_trajectories.reshape(
            batch_size * num_candidates,
            candidate_trajectories.shape[2],
            candidate_trajectories.shape[3],
        )
        candidate_latents = self._rollout_wm_future_latents(
            initial_latent=flat_keyval,
            trajectory=flat_trajectories,
            num_future_frames=self._wm_reward_num_future_frames,
            window_num_poses=self._wm_reward_window_num_poses,
            action_encoder=self._wm_reward_action_aware_encoder,
        )
        reward_feature = self._compute_wm_reward_feature(flat_keyval, candidate_latents)
        wm_candidate_logits = self._wm_reward_head(reward_feature).view(batch_size, num_candidates)

        output["wm_candidate_logits"] = wm_candidate_logits
        output["wm_candidate_base_logits"] = candidate_logits
        output["wm_candidate_indices"] = candidate_indices
        output["wm_candidate_trajectories"] = candidate_trajectories

    def _get_wm_candidate_selection_count(
        self,
        output: Dict[str, torch.Tensor],
        num_candidates: int,
    ) -> int:
        anchor_count = int(self._trajectory_head.plan_anchor.shape[0])
        if anchor_count <= 0:
            return num_candidates

        anchor_trajectories = output.get("anchor_trajectories")
        if anchor_trajectories is not None:
            goal_count = anchor_trajectories.shape[1] if anchor_trajectories.ndim == 6 else 1
        else:
            goal_count = max(1, num_candidates // anchor_count)

        return min(num_candidates, anchor_count * goal_count)

    def _compute_wm_reward_feature(
        self,
        initial_latent: torch.Tensor,
        wm_future_latents: torch.Tensor,
    ) -> torch.Tensor:
        all_latents = torch.cat([initial_latent[:, None], wm_future_latents], dim=1)
        batch_size, num_steps, _, channels = all_latents.shape

        bev_tokens = all_latents[:, :, :-1]
        bev_side = int(np.sqrt(bev_tokens.shape[2]))
        if bev_side * bev_side != bev_tokens.shape[2]:
            raise ValueError(f"Expected square BEV tokens, got {bev_tokens.shape[2]} tokens.")

        bev_feat_list = []
        for step_idx in range(num_steps):
            bev_feat = bev_tokens[:, step_idx].contiguous().view(batch_size, bev_side, bev_side, channels)
            bev_feat_list.append(bev_feat)
        all_bev_feature = torch.cat(bev_feat_list, dim=-1).permute(0, 3, 1, 2)
        reward_conv_output = self.reward_conv_net(all_bev_feature).squeeze(-1).permute(0, 2, 1)

        ego_feat_list = [all_latents[:, step_idx, -1:] for step_idx in range(num_steps)]
        cat_reward_feature = torch.cat(ego_feat_list + [reward_conv_output], dim=1)
        cat_reward_feature = cat_reward_feature.reshape(batch_size, -1)
        return self.reward_cat_head(cat_reward_feature)

    def _apply_wm_candidate_selection(self, output: Dict[str, torch.Tensor]) -> None:
        candidate_trajectories = output.get("wm_candidate_trajectories")
        wm_candidate_logits = output.get("wm_candidate_logits")
        base_logits = output.get("wm_candidate_base_logits")
        if candidate_trajectories is None or wm_candidate_logits is None or base_logits is None:
            return

        base_weight = float(getattr(self._config, "wm_base_logit_weight", 1.0))
        wm_weight = float(getattr(self._config, "wm_inference_score_weight", 1.0))
        base_scores = F.log_softmax(base_logits, dim=1)
        wm_scores = F.log_softmax(wm_candidate_logits, dim=1)
        final_scores = base_weight * base_scores + wm_weight * wm_scores
        best_index = final_scores.argmax(dim=1)
        gather_index = best_index[:, None, None, None].expand(
            -1,
            1,
            candidate_trajectories.shape[2],
            candidate_trajectories.shape[3],
        )
        output["trajectory"] = torch.gather(candidate_trajectories, dim=1, index=gather_index).squeeze(1)
        output["wm_candidate_base_scores"] = base_scores
        output["wm_candidate_scores"] = wm_scores
        output["wm_candidate_final_scores"] = final_scores
        output["wm_candidate_selected_index"] = best_index
        if "wm_candidate_indices" in output:
            output["wm_candidate_selected_candidate_index"] = torch.gather(
                output["wm_candidate_indices"],
                dim=1,
                index=best_index[:, None],
            ).squeeze(1)

    def _decode_wm_bev_semantic_map(self, wm_latent: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels = wm_latent.shape
        bev_tokens = wm_latent[:, :-1]
        bev_side = int(np.sqrt(bev_tokens.shape[1]))
        if bev_side * bev_side != bev_tokens.shape[1]:
            raise ValueError(f"Expected square BEV tokens, got {bev_tokens.shape[1]} tokens.")

        bev_feature = bev_tokens.permute(0, 2, 1).contiguous().view(batch_size, channels, bev_side, bev_side)
        bev_feature = F.interpolate(
            bev_feature,
            size=(
                self._config.lidar_resolution_height // self._config.bev_down_sample_factor,
                self._config.lidar_resolution_width // self._config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        bev_feature = self._wm_bev_feature_head(bev_feature)
        return self._bev_semantic_head(bev_feature)

    def _predict_next_latent(
        self,
        prev_latent: torch.Tensor,
        prev_trajectory: torch.Tensor,
        action_encoder: nn.Module,
    ) -> torch.Tensor:
        ego_trajectory_latent = self._encode_wm_ego_trajectory_feature(
            prev_latent=prev_latent,
            prev_trajectory=prev_trajectory,
            action_encoder=action_encoder,
        )
        trajectory_conditioned_latent = self._inject_wm_ego_trajectory_feature(
            prev_latent=prev_latent,
            ego_trajectory_latent=ego_trajectory_latent,
            prev_trajectory=prev_trajectory,
        )
        return self._wm_decoder(trajectory_conditioned_latent, trajectory_conditioned_latent)

    @staticmethod
    def _encode_wm_ego_trajectory_feature(
        prev_latent: torch.Tensor,
        prev_trajectory: torch.Tensor,
        action_encoder: nn.Module,
    ) -> torch.Tensor:
        batch_size = prev_latent.shape[0]
        prev_ego_latent = prev_latent[:, -1]
        prev_waypoints = prev_trajectory[..., :2].reshape(batch_size, -1)
        return action_encoder(torch.cat([prev_ego_latent, prev_waypoints], dim=-1))

    def _inject_wm_ego_trajectory_feature(
        self,
        prev_latent: torch.Tensor,
        ego_trajectory_latent: torch.Tensor,
        prev_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        bev_tokens = prev_latent[:, :-1]
        batch_size, num_bev_tokens, channels = bev_tokens.shape
        bev_side = int(np.sqrt(num_bev_tokens))
        if bev_side * bev_side != num_bev_tokens:
            raise ValueError(f"Expected square BEV tokens, got {num_bev_tokens} tokens.")

        endpoint = prev_trajectory[:, -1]
        bev_map = bev_tokens.permute(0, 2, 1).reshape(batch_size, channels, bev_side, bev_side)
        bev_map = self.inject_ego_feat_to_bev_map(
            bev_map=bev_map,
            new_features=ego_trajectory_latent,
            delta_x_y=endpoint[..., :2],
            H=bev_side,
            W=bev_side,
        )
        injected_bev_tokens = bev_map.permute(0, 2, 3, 1).reshape(batch_size, num_bev_tokens, channels)
        return torch.cat([injected_bev_tokens, ego_trajectory_latent[:, None]], dim=1)

    @staticmethod
    def inject_ego_feat_to_bev_map(
        bev_map: torch.Tensor,
        new_features: torch.Tensor,
        delta_x_y: torch.Tensor,
        H: int = 8,
        W: int = 8,
    ) -> torch.Tensor:
        batch_size, channels, height, width = bev_map.shape
        if height != H or width != W:
            raise ValueError(f"BEV map dimensions must be ({H}, {W}), but got ({height}, {width}).")
        if new_features.shape != (batch_size, channels):
            raise ValueError(f"new_features must have shape ({batch_size}, {channels}), got {new_features.shape}.")

        delta_x, delta_y = delta_x_y[:, 0], delta_x_y[:, 1]
        pixel_per_meter_x = H / 32.0
        pixel_per_meter_y = W / 64.0

        h_idx = delta_x * pixel_per_meter_x
        w_idx = delta_y * pixel_per_meter_y + (W / 2.0)

        h0 = torch.floor(h_idx).long()
        w0 = torch.floor(w_idx).long()
        h1 = h0 + 1
        w1 = w0 + 1

        dh = h_idx - h0.float()
        dw = w_idx - w0.float()
        weights = torch.stack(
            [
                (1 - dh) * (1 - dw),
                (1 - dh) * dw,
                dh * (1 - dw),
                dh * dw,
            ],
            dim=1,
        )
        h_indices = torch.stack([h0, h0, h1, h1], dim=1)
        w_indices = torch.stack([w0, w1, w0, w1], dim=1)
        batch_indices = torch.arange(batch_size, device=bev_map.device).view(batch_size, 1).repeat(1, 4)

        h_indices_flat = h_indices.reshape(-1)
        w_indices_flat = w_indices.reshape(-1)
        weights_flat = weights.reshape(-1)
        batch_indices_flat = batch_indices.reshape(-1)

        valid = (
            (h_indices_flat >= 0)
            & (h_indices_flat < H)
            & (w_indices_flat >= 0)
            & (w_indices_flat < W)
        )
        if not valid.any():
            return bev_map

        h_indices_valid = h_indices_flat[valid]
        w_indices_valid = w_indices_flat[valid]
        weights_valid = weights_flat[valid].to(dtype=new_features.dtype).unsqueeze(1)
        batch_indices_valid = batch_indices_flat[valid]

        weighted_features = (new_features[batch_indices_valid] * weights_valid).to(dtype=bev_map.dtype)
        channel_indices = torch.arange(channels, device=bev_map.device).view(1, channels).repeat(
            weighted_features.shape[0],
            1,
        )
        linear_indices = (
            batch_indices_valid.unsqueeze(1) * channels * H * W
            + channel_indices * H * W
            + h_indices_valid.unsqueeze(1) * W
            + w_indices_valid.unsqueeze(1)
        ).reshape(-1)

        bev_map_flat = bev_map.reshape(-1).clone()
        bev_map_flat.index_add_(0, linear_indices, weighted_features.reshape(-1))
        return bev_map_flat.view(batch_size, channels, H, W)

    def _get_wm_trajectory_window(
        self,
        trajectory: torch.Tensor,
        window_start: int,
        window_num_poses: int,
    ) -> torch.Tensor:
        window_end = window_start + window_num_poses
        if window_end > trajectory.shape[1]:
            raise ValueError(
                f"WM trajectory window [{window_start}, {window_end}) exceeds predicted path length "
                f"{trajectory.shape[1]}."
            )

        window = trajectory[:, window_start:window_end]
        if window_start == 0:
            return window

        origin = trajectory[:, window_start - 1:window_start]
        dx = window[..., StateSE2Index.X] - origin[..., StateSE2Index.X]
        dy = window[..., StateSE2Index.Y] - origin[..., StateSE2Index.Y]
        dtheta = window[..., StateSE2Index.HEADING] - origin[..., StateSE2Index.HEADING]

        cos_theta = torch.cos(origin[..., StateSE2Index.HEADING])
        sin_theta = torch.sin(origin[..., StateSE2Index.HEADING])

        rebased_x = cos_theta * dx + sin_theta * dy
        rebased_y = -sin_theta * dx + cos_theta * dy
        rebased_heading = self._normalize_angle(dtheta)
        return torch.stack([rebased_x, rebased_y, rebased_heading], dim=-1)

    @staticmethod
    def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        # Mimir keeps image-projected navigation features in the navi attention path.
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        if config.use_unc_score == True:
            self.cross_bev_attention_navi = GridSampleCrossBEVAttention_naviscore(
                config.tf_d_model,
                config.tf_num_head,
                num_points=1,
                config=config,
                in_bev_dims=256,
            )
        else:
            self.cross_bev_attention_navi = GridSampleCrossBEVAttention_navi(
                config.tf_d_model,
                config.tf_num_head,
                num_points=1,
                config=config,
                in_bev_dims=256,
            )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None,
                navi_points=None,
                points_score=1.0):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        if navi_points is not None:
            traj_feature = self.cross_bev_attention_navi(
                traj_feature,
                navi_points,
                bev_feature,
                bev_spatial_shape,
                points_score,
            )
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=None)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None,
                navi_points=None,
                points_score=1.0):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
                navi_points,
                points_score=points_score,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class DDIMScheduler_with_logprob(DDIMScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0, # 1.0 for ddpm, 0.0 for ddim
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // self.num_inference_steps
        )
        # # to prevent OOB on gather
        # prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_(min=0) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample_mean is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
                " `prev_sample` stays `None`."
            )

        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        if prev_sample is None:
            # 乘性噪声
            variance_noise_horizon = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon,variance_noise_vert),dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1,1,model_output.shape[2],1)

            # 加性噪声
            variance_noise_x = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0],model_output.shape[1],1,1], generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_add = torch.cat((variance_noise_x,variance_noise_y),dim=-1)
            variance_noise_add = variance_noise_add.repeat(1,1,model_output.shape[2],1)

            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add

        std_dev_t_mul = torch.clip(std_dev_t, min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )   

        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: MimirConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._config = config
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.num_groups = config.num_groups
        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.goalpoint_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,256),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        self.sigmoid = nn.Sigmoid()
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
        self.use_gt_goal_train = config.use_gt_goal_train
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.targets = [] 
        self.num_draw = 0
        self._score_buf = []
        # pdm score
        pdm_cfg = OmegaConf.load('navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')
        self.simulator_cfg = pdm_cfg.simulator
        self.scorer_cfg = pdm_cfg.scorer

        self._pdm_pool = cf.ProcessPoolExecutor(
            max_workers=24,
            mp_context=mp.get_context("spawn"),
            initializer=_init_pool,
            initargs=(self.simulator_cfg, self.scorer_cfg),
        )
        self.metric_caches = {}
        self.simulator: PDMSimulator = instantiate(self.simulator_cfg)
        self.scorer: PDMScorer = instantiate(self.scorer_cfg)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)


    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,eta=0.0, old_pred=None,metric_cache=None, cal_pdm=True,token=None,goalpoint=None,points_score=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            if old_pred is not None:
                return self.get_rlloss(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,eta,old_pred,goalpoint=goalpoint,points_score=points_score)
            else:
                return self.forward_train_rl(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,eta,metric_cache,cal_pdm=cal_pdm,token=token,goalpoint=goalpoint,points_score=points_score)
        else:
            return self.forward_test_rl(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,metric_cache,eta=0.0,goalpoint=goalpoint,points_score=points_score)

    def get_pdm_score_para(self, trajectory, metric_cache_path):
        if metric_cache_path is None:
            raise RuntimeError(
                "Mimir RL training requires PDM metric cache paths. "
                "Use cache features with metric_cache_path or pass metric_cache into agent.forward."
            )
        B, G = trajectory.shape[:2]
        if isinstance(metric_cache_path, (str, os.PathLike)):
            metric_cache_path = [metric_cache_path] * B
        traj_np = trajectory.detach().cpu().numpy()
        futures = [
            self._pdm_pool.submit(
                _pdm_worker,
                (metric_cache_path[b], traj_np[b]),
            )
            for b in range(B)
        ]
        scores_np = np.vstack([f.result()[0] for f in futures])    # (B,G)
        metric_cache = [f.result()[1] for f in futures]
        sub_scores  = [f.result()[2] for f in futures]
        return torch.from_numpy(scores_np).to(trajectory.device), metric_cache, sub_scores

    def _prepare_goal_inputs(self, goalpoint, points_score, bs, device, dtype):
        if goalpoint is None:
            goalpoint = torch.zeros((bs, 1, 2), device=device, dtype=dtype)
        elif goalpoint.ndim != 3 or goalpoint.shape[0] != bs or goalpoint.shape[-1] != 2:
            raise ValueError(f"Expected goalpoint shape (B, G, 2), got {goalpoint.shape}.")
        else:
            goalpoint = goalpoint.to(device=device, dtype=dtype)

        if points_score is None:
            points_score = torch.zeros_like(goalpoint)
        elif points_score.shape != goalpoint.shape:
            raise ValueError(f"Expected points_score shape {goalpoint.shape}, got {points_score.shape}.")
        else:
            points_score = points_score.to(device=device, dtype=dtype)
        return goalpoint, points_score

    @staticmethod
    def _expand_by_goal(tensor, goal_count):
        if goal_count == 1:
            return tensor
        bs, num_modes = tensor.shape[:2]
        return tensor[:, None].expand(-1, goal_count, *([-1] * (tensor.ndim - 1))).reshape(
            bs,
            goal_count * num_modes,
            *tensor.shape[2:],
        )

    @staticmethod
    def _build_per_query_goal_inputs(goalpoint, points_score, modes_per_goal):
        bs, goal_count, _ = goalpoint.shape
        query_goalpoint = goalpoint[:, :, None, :].expand(-1, -1, modes_per_goal, -1)
        query_goalpoint = query_goalpoint.reshape(bs, goal_count * modes_per_goal, 1, 2)
        query_points_score = points_score[:, :, None, :].expand(-1, -1, modes_per_goal, -1)
        query_points_score = query_points_score.reshape(bs, goal_count * modes_per_goal, 1, 2)
        return query_goalpoint, query_points_score

    def _add_goal_condition(self, traj_feature, bs, ego_fut_mode, goalpoint=None, points_score=None):
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        if goalpoint is None:
            return traj_feature
        if points_score is None:
            points_score = torch.ones_like(goalpoint)
        goal_info_embed = gen_sineembed_for_position(
            torch.stack([goalpoint.squeeze(2), points_score.squeeze(2)], dim=2),
            hidden_dim=128,
        )
        goal_info_embed = goal_info_embed.flatten(-2)
        goal_info_feature = self.goalpoint_encoder(goal_info_embed)
        return traj_feature + goal_info_feature.view(bs, ego_fut_mode, -1)

    def forward_train_rl(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets,global_img, eta,metric_cache,cal_pdm,token,goalpoint=None,points_score=None) -> Dict[str, torch.Tensor]:
        step_num = 10
        bs = ego_query.shape[0]
        device = ego_query.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        num_groups = self.num_groups
        modes_per_goal = num_groups * self.ego_fut_mode
        
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs,num_groups, 1, 1, 1)  

        plan_anchor = plan_anchor.view(bs, modes_per_goal, *plan_anchor.shape[3:]) # bs num_groups * 20, 8, 2
        plan_anchor = self._expand_by_goal(plan_anchor, goal_count)
        diffusion_output = self.norm_odo(plan_anchor)
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(goalpoint, points_score, modes_per_goal)
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0

        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusionrl_scheduler.add_noise(original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)

        all_log_probs = []
        all_diffusion_output= [diffusion_output]

        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_condition(
                traj_feature,
                bs,
                diffusion_output.shape[1],
                goalpoint=query_goalpoint,
                points_score=query_points_score,
            )

            timesteps = k.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed, status_encoding, global_img,
                query_goalpoint,
                points_score=decoder_points_score,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[..., :2] # bs G*N 8 2
            x_start = self.norm_odo(x_start)
            # get prev_sample
            prev_sample, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=eta,
            )
            diffusion_output = prev_sample
            all_log_probs.append(log_prob)
            all_diffusion_output.append(prev_sample) # BG N 8 2

        all_log_probs = torch.stack(all_log_probs, dim=-1) # B G*N step_num
        # BG N step_num
        all_log_probs = all_log_probs.view(bs, goal_count, num_groups, self.ego_fut_mode, all_log_probs.shape[-1])  # B P G N step_num
        all_diffusion_output = torch.stack(all_diffusion_output, dim=-1) # BG N step_num

        diffusion_output = self.denorm_odo(diffusion_output) # B G*N 8 2
        diffusion_output = self.bezier_xyyaw(diffusion_output)

        target_traj = targets['trajectory'].unsqueeze(1)
        diffusion_output_with_gt = torch.cat((diffusion_output,target_traj),dim=1)
        candidate_logits = None
        if cal_pdm:
            reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(diffusion_output_with_gt, metric_cache)      # (B,G)
            reward_gt = reward_group[:, -1:].view(bs, 1, 1, 1)
            reward_group = reward_group[:,:-1]  # remove the last group which is GT  
            candidate_logits = reward_group

            #sub score
            keys = sub_rewards_group[0].keys()
            batched_sub = {
                k: torch.as_tensor(
                        np.vstack([d[k] for d in sub_rewards_group]),  # (B, G)
                        device=reward_group.device, dtype=reward_group.dtype
                    )
                for k in keys
            }

            # 逐anchor
            reward_group = reward_group.view(bs, goal_count, num_groups, self.ego_fut_mode)  # (B,P,G,N)
            mean_grouped_rewards = reward_group.mean(dim=(1, 2))
            std_grouped_rewards = reward_group.std(dim=(1, 2))
            advantages = (
                reward_group - mean_grouped_rewards[:, None, None, :]
            ) / (std_grouped_rewards[:, None, None, :] + 1e-4)

            # 只保留 “好于 GT” 的正向样本
            mask_positive = (reward_group > (reward_gt-1e-6))                         # (B,G) bool
            advantages = advantages.clamp(min=0) * mask_positive.float()       # 负 adv 归 0

            # 根据sub reward来调节adv
            for k, v_full in batched_sub.items():
                v = v_full[:, :-1]                                       # 去掉最后一列 GT  → (B, G-1)
                v = v.view(bs, goal_count, num_groups, self.ego_fut_mode)
                
                if k == 'no_collision' or k == 'drivable_area':
                    zero_mask = (v != 1)
                    advantages = torch.where(zero_mask, torch.full_like(advantages, -1.0), advantages)
                else:  # 'ttc', 'comfort', 'final', 'dir_weighted', 'progress'
                    continue

            # for log
            pos_cnt  = mask_positive.sum(dim=(1,2,3), keepdim=True)                  # (B,1)
            pos_sum  = (reward_group * mask_positive.float()).sum(dim=(1,2,3), keepdim=True)

            mean_pos = pos_sum / pos_cnt.clamp(min=1)
            mean_all = reward_group.mean(dim=(1,2,3), keepdim=True)

            batch_reward = torch.where(pos_cnt > 0, mean_pos, mean_all)   # (B,1)
            reward = batch_reward.squeeze(-1).mean()                      # for log

            #去掉 GT 列，并套用与 reward_group 相同的 mask
            sub_rewards_mean = {}
            for k, v_full in batched_sub.items():
                v = v_full[:, :-1]
                gt_k   = v_full[:,  -1:]
                v = v.view(bs, goal_count, num_groups, self.ego_fut_mode)
                mean_all_k = v.mean(dim=(1, 2), keepdim=True)
                sub_rewards_mean[k] = mean_all_k.mean().item()
            advantages = advantages.view(bs, goal_count*num_groups*self.ego_fut_mode)
            advantages = advantages.detach().unsqueeze(-1).repeat(1,1,step_num)
            discount = torch.tensor(
                [
                    0.8 ** (step_num - i - 1)
                    for i in range(step_num)
                ]
            ).to(advantages.device)
            advantages = advantages * discount

        else:
            advantages = None
            reward = None
            sub_rewards_mean = None
            candidate_logits = poses_cls

        best_idx = candidate_logits.argmax(dim=-1)
        best_traj = diffusion_output[torch.arange(bs, device=device), best_idx]
        return {
            "trajectory": best_traj,
            "candidate_trajectories": diffusion_output,
            "candidate_logits": candidate_logits,
            "all_diffusion_output": all_diffusion_output,
            "advantages": advantages,
            "reward": reward,
            "sub_rewards": sub_rewards_mean,
        }


    def forward_test_rl(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets,global_img,metric_cache,eta=0.0,goalpoint=None,points_score=None) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = self.num_groups
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs,num_groups,1,1,1)
        # plan_anchor = plan_anchor[:,:,16:17].repeat(1, 1, self.ego_fut_mode, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])
        modes_per_goal = plan_anchor.shape[1]
        plan_anchor = self._expand_by_goal(plan_anchor, goal_count)
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(goalpoint, points_score, modes_per_goal)
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
        all_diffusion_output = [diffusion_output]
        all_log_probs = []
        ego_fut_mode = diffusion_output.shape[1]
        for i, k in enumerate(roll_timesteps[:]):
            # diffusion_output_xy = diffusion_output[..., :2]  # 只保留 x, y
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_condition(
                traj_feature,
                bs,
                ego_fut_mode,
                goalpoint=query_goalpoint,
                points_score=query_points_score,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
                query_goalpoint,
                points_score=decoder_points_score,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            # x_start = poses_reg
            x_start = self.norm_odo(x_start)
            diffusion_output,log_prob,diffusion_output_mean = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=0.0,
            )
            all_diffusion_output.append(diffusion_output)
            all_log_probs.append(log_prob)

        diffusion_output = self.denorm_odo(diffusion_output_mean)

        diffusion_output = self.bezier_xyyaw(diffusion_output)  # (B, G, 8, 1)

        all_diffusion_output = torch.stack(all_diffusion_output, dim=-1) # BG N step_num
        all_log_probs = torch.stack(all_log_probs, dim=-1) # BG N step_num


        if metric_cache is None:
            best_idx = poses_cls.argmax(dim=-1)
            best_traj = diffusion_output[torch.arange(bs, device=device), best_idx]
            return {
                "trajectory": best_traj,
                "candidate_trajectories": diffusion_output,
                "candidate_logits": poses_cls,
                "all_diffusion_output": all_diffusion_output,
                "log_probs": all_log_probs,
            }

        reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(diffusion_output, metric_cache)
        candidate_logits = reward_group

        best_idx = reward_group.argmax(dim=-1)
        best_traj = diffusion_output[torch.arange(bs, device=device), best_idx]
        reward_group = reward_group.max(dim=-1)[0]

        target_traj = targets['trajectory'].unsqueeze(1)
        trajectory_loss = F.l1_loss(diffusion_output, target_traj)

        keys = sub_rewards_group[0].keys()
        sub_rewards_group = {
            k: np.vstack([d[k] for d in sub_rewards_group])    # 形状 (B, 1)
            for k in keys
        }
        sub_scores_mean = {
            k: v.mean().item()    # .item() 把 0-D ndarray 转成 Python float
            for k, v in sub_rewards_group.items()
        }
        return {
            "trajectory": best_traj,
            "candidate_trajectories": diffusion_output,
            "candidate_logits": candidate_logits,
            "loss": trajectory_loss,
            "reward": reward_group.mean(),
            "sub_rewards": sub_scores_mean,
            "all_diffusion_output": all_diffusion_output,
            "log_probs": all_log_probs,
        }

    def get_rlloss(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets,global_img, eta, old_pred,goalpoint=None,points_score=None):

        old_diffusion_output = old_pred['all_diffusion_output']
        advantages = old_pred['advantages']

        chains = old_diffusion_output[...,:-1]
        chains_prev = old_diffusion_output[...,1:]  

        step_num = 10
        bs = chains.shape[0]
        device = chains.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        all_log_probs = []
        poses_reg_steps_list = []
        poses_cls_steps_list = []
        for i, k in enumerate(roll_timesteps[:]):
            diffusion_output = chains[..., i]
            ego_fut_mode = diffusion_output.shape[1]
            modes_per_goal = ego_fut_mode // goal_count
            query_goalpoint, query_points_score = self._build_per_query_goal_inputs(goalpoint, points_score, modes_per_goal)
            decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_condition(
                traj_feature,
                bs,
                ego_fut_mode,
                goalpoint=query_goalpoint,
                points_score=query_points_score,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed, status_encoding, global_img,
                query_goalpoint,
                points_score=decoder_points_score,
            )
            poses_reg_steps_list.append(poses_reg_list)
            poses_cls_steps_list.append(poses_cls_list)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[..., :2] # bs G*N 8 2
            # x_start = poses_reg
            x_start = self.norm_odo(x_start)
            _, log_prob, _ = self.diffusionrl_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=diffusion_output,
                eta=eta,
                prev_sample=chains_prev[...,i]
            )
            all_log_probs.append(log_prob)
        all_log_probs = torch.stack(all_log_probs, dim=-1) # BG N step_num
        per_token_logps = all_log_probs.view(bs, -1, all_log_probs.shape[-1])  # B P*G*N step_num

        per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages

        # ---------- (1) RL 损失，保留 batch 维 ----------
        mask_nz    = per_token_loss != 0            # (B,G,T)
        RL_loss_b  = (per_token_loss * mask_nz).sum(dim=1) \
                    / mask_nz.sum(dim=1).clamp_min(1)      # (B,T)
        RL_loss_b = RL_loss_b.mean(dim=-1)  # (B,)
        # ---------- (2) IL 损失，先算每个 batch ----------
        IL_loss_b = torch.zeros_like(RL_loss_b)     # (B,)
        target_traj = targets['trajectory'].unsqueeze(1).repeat(1, per_token_logps.shape[1], 1, 1)
        
        for poses_reg_list in poses_reg_steps_list:                 # 5 个 time-step
            for poses_reg in poses_reg_list:                        # 2 层 decoder
                traj_l1 = F.l1_loss(poses_reg[...,:2], target_traj[...,:2], reduction='none')  # (B,G,T,C)
                IL_loss_b += traj_l1.mean()                   # 加到 (B,)

        IL_loss_b /= (len(poses_reg_steps_list) * len(poses_reg_steps_list[0]))  # 取平均
        has_positive   = (advantages > 0).any(dim=2).any(dim=1)         # (B,) bool

        il_weight  = torch.where(has_positive == 0,          # (B,)
                                torch.tensor(1.0,  device=RL_loss_b.device),
                                torch.tensor(0.1, device=RL_loss_b.device))
        loss_b     = RL_loss_b + il_weight*IL_loss_b  # (B,)
        loss       = loss_b.mean()                        # 标量
        output = {"loss": loss}
        for key in ("trajectory", "candidate_trajectories", "candidate_logits"):
            if key in old_pred:
                output[key] = old_pred[key]
        return output


    def bezier_xyyaw(self,xy8: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        xy8 : Tensor, shape = (B, G, 8, 2)
            仅包含未来 8 个 (x, y) 预测点，默认以 (0,0) 为局部坐标系原点
        Returns
        -------
        xyyaw : Tensor, shape = (B, G, 8, 3)
                对应 8 个预测点的 (x, y, yaw)（弧度）
        """
        assert xy8.shape[-2:] == (8, 2), "Input must be (B,G,8,2)"
        B, G, _, _ = xy8.shape
        device, dtype = xy8.device, xy8.dtype

        # ---------- ①  在最前面插入固定起点 (0,0) ----------
        origin = torch.zeros_like(xy8[..., :1, :])      # (B,G,1,2)
        ctrl   = torch.cat([origin, xy8], dim=-2)       # (B,G,9,2)
        n      = ctrl.shape[-2] - 1                     # 8 阶 Bézier

        # ΔP_i = P_{i+1} - P_i  → (B,G,8,2)
        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]

        # 组合数 C(n-1,i),  i = 0…7
        binom = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)],
            device=device, dtype=dtype
        )                                               # (8,)

        # ---------- ②  采样 t_k = k / n ,  k = 1…8 ----------
        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n   # (8,)

        # Bernstein 基函数 (一阶导数用)  → (8,8)
        t_pow   = t.view(-1, 1) ** torch.arange(0, n,     device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n-1, -1, -1, device=device, dtype=dtype)
        basis   = binom * t_pow * one_pow

        # 扩维广播
        delta_exp = delta.unsqueeze(2)                  # (B,G,1,8,2)
        basis_exp = basis.view(1, 1, 8, 8, 1)           # (1,1,8,8,1)

        # 一阶导：B'(t_k) = n * Σ_i basis_i(t_k) * ΔP_i
        deriv = n * (delta_exp * basis_exp).sum(dim=3)  # (B,G,8,2)

        # yaw = atan2(dy, dx)
        dx, dy = deriv[..., 0], deriv[..., 1]
        yaw = torch.atan2(dy, dx).unsqueeze(-1)         # (B,G,8,1)

        return torch.cat([xy8, yaw], dim=-1)            # (B,G,8,3)