from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import lzma
import pickle
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.mimir_grpo.mimir_config import MimirConfig
from navsim.agents.mimir_grpo.mimir_backbone import MimirBackbone
from navsim.agents.mimir_grpo.mimir_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.mimir_grpo.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.mimir_grpo.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention,GridSampleCrossBEVAttention_navi,GridSampleCrossBEVAttention_naviscore
from navsim.agents.mimir_grpo.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from torch.distributions import Normal
from typing import Any, List, Dict, Optional, Union,Tuple
import numpy.typing as npt
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def _transform_navi_to_camera_tensor(
    navi: torch.Tensor,                     # (N, 3)
    sensor2lidar_rotation: torch.Tensor,   # (3, 3)
    sensor2lidar_translation: torch.Tensor # (3,)
) -> torch.Tensor:
    """
    将导航点从 VCS 坐标变换到摄像机坐标系
    """
    lidar2cam_r = sensor2lidar_rotation.inverse()
    lidar2cam_t = -torch.matmul(lidar2cam_r, sensor2lidar_translation)

    locs_homo = torch.cat([navi, torch.ones_like(navi[:, :1])], dim=-1)  # (N, 4)

    # 构建 4x4 齐次变换矩阵
    lidar2cam_rt = torch.eye(4, device=navi.device).to(navi)
    lidar2cam_rt[:3, :3] = lidar2cam_r
    lidar2cam_rt[:3, 3] = lidar2cam_t

    locs_cam = torch.matmul(lidar2cam_rt, locs_homo.T).T[:, :3]  # (N, 3)
    return locs_cam

def _project_points_to_image_tensor(
    points: torch.Tensor,           # (N, 3) in camera frame
    intrinsics: torch.Tensor,      # (3, 3)
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将摄像机系下的点投影到图像像素平面
    返回 (N, 2) 像素位置 + mask 是否在图像有效区域
    """
    N = points.shape[0]
    pc_homo = torch.cat([points, torch.ones((N, 1), device=points.device,dtype=points.dtype)], dim=-1)  # (N, 4)
    intrinsics_pad = torch.eye(4, device=points.device,dtype=points.dtype)
    intrinsics_pad[:3, :3] = intrinsics

    proj = torch.matmul(pc_homo, intrinsics_pad.T)  # (N, 4)
    z = proj[:, 2:3].clamp(min=eps)
    xy = proj[:, 0:2] / z  # (N, 2)

    in_fov = proj[:, 2] > eps

    if image_shape is not None:
        H, W = image_shape
        u, v = xy[:, 0], xy[:, 1]
        in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        in_fov = in_fov & in_bounds

    return xy, in_fov

def extract_feature_values_at_navi_batched(
    feature_map: torch.Tensor,                 # (B, C_feat, H_feat, W_feat)
    navi_tensor: torch.Tensor,                 # (B, N, 3)
    sensor2lidar_rotation: torch.Tensor,       # (B, 3, 3)
    sensor2lidar_translation: torch.Tensor,    # (B, 3)
    intrinsics: torch.Tensor,                  # (B, 3, 3)
    image_shape: Tuple[int, int]               # 原始图像尺寸 (H_img, W_img)
) -> torch.Tensor:
    B, C_feat, H_feat, W_feat = feature_map.shape
    _, N, _ = navi_tensor.shape
    H_img, W_img = image_shape
    device = navi_tensor.device

    feature_values = torch.zeros((B, C_feat, N), device=device)

    for b in range(B):
        # 单图提取
        navi_cam = _transform_navi_to_camera_tensor(
            navi_tensor[b], sensor2lidar_rotation[b], sensor2lidar_translation[b]
        )
        pixel_coords, valid_mask = _project_points_to_image_tensor(
            navi_cam, intrinsics[b], image_shape=image_shape
        )
        # 缩放到特征图分辨率
        pixel_coords_scaled = pixel_coords.clone()
        pixel_coords_scaled[:, 0] *= W_feat / W_img
        pixel_coords_scaled[:, 1] *= H_feat / H_img

        u = pixel_coords_scaled[:, 0].round().long().clamp(0, W_feat - 1)
        v = pixel_coords_scaled[:, 1].round().long().clamp(0, H_feat - 1)

        feature_values[b] = feature_map[b, :, v, u] * valid_mask.unsqueeze(0)

    return feature_values  # shape: (B, C_feat, N)

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
        super(RewardConvNet, self).__init__()
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


class MimirModel(nn.Module):
    """Torch module for Mimir."""

    def __init__(self, config: MimirConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
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
            self._status_encoding=nn.Linear(4+1+1,config.tf_d_model)
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
        self.weight_score=None
        if self._config.unc_path:
            self.weight_score=np.load(self._config.unc_path,allow_pickle=True).item()

        self.goalpoints=None
        if self._config.navi_path:
            self.goalpoints=np.load(self._config.navi_path,allow_pickle=True).item()

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

    def init_grpo_reference(self) -> None:
        """Keep an IL-aligned frozen trajectory head for GRPO behavior cloning."""
        self._trajectory_head.old_policy = copy.deepcopy(self._trajectory_head)
        self._trajectory_head.old_policy.old_policy = None
        self._trajectory_head.old_policy.eval()
        for param in self._trajectory_head.old_policy.parameters():
            param.requires_grad = False

    def set_frozen_modules_to_eval_mode(self) -> None:
        if not (self.training and getattr(self._config, "grpo", False)):
            return

        frozen_modules = [
            self._backbone,
            self._keyval_embedding,
            self._query_embedding,
            self._bev_downscale,
            self._status_encoding,
            self._bev_semantic_head,
            self._tf_decoder,
            self._agent_head,
            self.bev_proj,
        ]
        if getattr(self._config, "use_wm", False):
            frozen_modules.extend(
                [
                    self._wm_decoder,
                    self._wm_action_aware_encoder,
                    self._wm_reward_action_aware_encoder,
                    self._wm_bev_feature_head,
                    self.reward_conv_net,
                    self.reward_cat_head,
                    self._wm_reward_head,
                ]
            )
        for module in frozen_modules:
            module.eval()
        if self._trajectory_head.old_policy is not None:
            self._trajectory_head.old_policy.eval()

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None,goalpoint=None,token=None,metric_cache_paths=None,grpo: bool = False) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        self.set_frozen_modules_to_eval_mode()

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        if self._config.status_norm and self._config.training==False:
            vle = torch.norm(status_feature[:,4:6],dim=-1,keepdim=True)
            acc = torch.norm(status_feature[:,6:8],dim=-1,keepdim=True)
            dot_product = torch.sum(status_feature[:,4:6]*status_feature[:,6:8],dim=-1,keepdim=True)
            tag_vle = torch.where(status_feature[:,4:5] >= 0,1.0,-1.0)
            tag_acc = torch.where(dot_product > 0,1.0,-1.0)
            status_feature= torch.cat([status_feature[:,:4], tag_vle*vle,tag_acc*acc], dim=-1)
        
        
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, img_feature = self._backbone(camera_feature, lidar_feature)
        if self._config.training and self._config.use_proj_image:
            navis=features['gt_trajs'][:,-1:,:]
            navis[:,:,2]=0.0
            R=features['sensor2lidar_rot'][:,0].to(navis)
            T=features['sensor2lidar_trans'][:,0].to(navis)
            K=features['intrinsic'][:,0].to(navis)
            pix=extract_feature_values_at_navi_batched(img_feature[:,:,:,17:-17],navis,R,T,K,(1024,1920))
        elif self._config.use_proj_image:
            R=features['sensor2lidar_rot'][:,0].to(camera_feature)
            T=features['sensor2lidar_trans'][:,0].to(camera_feature)
            K=features['intrinsic'][:,0].to(camera_feature)
            goalpoint_tensor=torch.from_numpy(goalpoint).to(camera_feature).unsqueeze(0).unsqueeze(0)
            zeros=torch.zeros_like(goalpoint_tensor)
            goalpoint_tensor=torch.cat([goalpoint_tensor,zeros[:,:,-1:]],dim=-1)
            pix=extract_feature_values_at_navi_batched(img_feature[:,:,:,17:-17],goalpoint_tensor,R,T,K,(1024,1920))
        else:
            pix=None
        # pix.shape [bs,256,1]

        if self.weight_score:
            if self._config.training:
                points_score=load_navis_from_np(self.weight_score,features['token'],self._config.num_goal_points).to(bev_feature)
            else:
                points_score=load_navis_from_np(self.weight_score,token,self._config.num_goal_points).to(bev_feature)
        else:
            points_score=None
        
        if self.goalpoints:
            if self._config.training:
                goalpoint=load_navis_from_np(self.goalpoints,features['token'],self._config.num_goal_points).to(bev_feature)
            else:
                goalpoint=load_navis_from_np(self.goalpoints,token,self._config.num_goal_points).to(bev_feature)
        else:
            goalpoint=None
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

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        if grpo:
            trajectory = self._trajectory_head.forward_grpo(
                trajectory_query,
                agents_query,
                cross_bev_feature,
                bev_spatial_shape,
                status_encoding[:, None],
                targets=targets,
                global_img=pix,
                goalpoint=goalpoint,
                points_score=points_score,
                metric_cache_paths=metric_cache_paths,
                wm_selector=self._select_sample_with_wm if self._config.use_wm else None,
                wm_keyval=keyval if self._config.use_wm else None,
            )
            return trajectory

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=pix,goalpoint=goalpoint,points_score=points_score)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        if self._config.use_wm:
            self._add_wm_candidate_selection_outputs(output, keyval)
            self._apply_wm_candidate_selection(output)

            if targets is not None:
                if "wm_future_bev_semantic_map" not in targets:
                    target_keys = ", ".join(sorted(targets.keys()))
                    raise RuntimeError(
                        "use_wm=True requires future BEV semantic map targets, but none were found. "
                        "Rebuild the cache with the WM future BEV target builder, "
                        f"and check target keys. Received target keys: {target_keys}"
                    )

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
        """Predict future ego-frame BEV maps from latent WM rollout."""

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
        """Roll out future latent scene tokens conditioned by a candidate trajectory."""

        wm_latent = initial_latent
        wm_future_latents = []

        for future_offset in range(1, num_future_frames + 1):
            window_start = future_offset - 1
            wm_trajectory = self._get_wm_trajectory_window(
                trajectory=trajectory,
                window_start=window_start,
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

    def _select_sample_with_wm(
        self,
        sampled: Dict[str, torch.Tensor],
        keyval: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        candidate_trajectories = sampled.get("candidate_trajectories")
        candidate_logits = sampled.get("candidate_logits")
        if candidate_trajectories is None or candidate_logits is None:
            return sampled

        sample_batch = candidate_logits.shape[0]
        if keyval.shape[0] == sample_batch:
            wm_keyval = keyval
        elif sample_batch % keyval.shape[0] == 0:
            wm_keyval = keyval.repeat_interleave(sample_batch // keyval.shape[0], 0)
        else:
            raise ValueError(
                f"Cannot align WM keyval batch {keyval.shape[0]} with GRPO sample batch {sample_batch}."
            )

        output: Dict[str, torch.Tensor] = {
            "trajectory": sampled["trajectory"],
            "candidate_trajectories": candidate_trajectories,
            "candidate_logits": candidate_logits,
        }
        if sampled.get("anchor_trajectories") is not None:
            output["anchor_trajectories"] = sampled["anchor_trajectories"]

        self._add_wm_candidate_selection_outputs(output, wm_keyval)
        self._apply_wm_candidate_selection(output)
        if "wm_candidate_selected_candidate_index" not in output:
            return sampled

        reranked = dict(sampled)
        reranked["trajectory"] = output["trajectory"].detach()
        reranked["selected_candidate_index"] = output["wm_candidate_selected_candidate_index"].detach()
        for key in (
            "wm_candidate_base_scores",
            "wm_candidate_scores",
            "wm_candidate_final_scores",
            "wm_candidate_selected_index",
        ):
            if key in output:
                reranked[key] = output[key].detach()
        return reranked

    def _add_wm_candidate_selection_outputs(
        self,
        output: Dict[str, torch.Tensor],
        keyval: torch.Tensor,
    ) -> None:
        """Score all trajectory candidates with a WoTE-style WM reward head."""

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
        """Use the training-time candidate count as the max number of WM-scored candidates."""

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
        """Build WoTE-style reward features from current and predicted future latents."""

        all_latents = torch.cat([initial_latent[:, None], wm_future_latents], dim=1)
        batch_size, num_steps, num_tokens, channels = all_latents.shape

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
        """Decode predicted WM latent tokens into a future ego-frame BEV semantic map."""

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
        """Predict next ego-frame latent tokens after injecting ego/trajectory feature into BEV tokens."""
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
        """Take a path slice and express it in the previous predicted ego frame."""
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
        # if global_img is not None:
        #     if len(global_img.shape)==4:
        #         global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
        #     else:
        #         global_img = global_img.permute(0,2,1).contiguous()
        #     global_feature = torch.cat([
        #             global_img, global_feature
        #         ], axis=-1)
        
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

        # GridSampleCrossBEVAttention_navi
        if config.use_unc_score==True:
            self.cross_bev_attention_navi= GridSampleCrossBEVAttention_naviscore(
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
                points_score=1.0,
                ):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        if navi_points is not None:
            traj_feature = self.cross_bev_attention_navi(traj_feature,navi_points,bev_feature,bev_spatial_shape,points_score)
        
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
                points_score=1.0,
                ):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,points_score=points_score)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

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

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
        self.use_gt_goal_train=config.use_gt_goal_train
        self.old_policy = None

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

    def _encode_goal_condition(self, goalpoint, points_score):
        goal_condition = torch.stack([goalpoint, points_score], dim=2)
        goal_info_embed = gen_sineembed_for_position(goal_condition, hidden_dim=128)
        goal_info_embed = goal_info_embed.flatten(-2)
        return self.goalpoint_encoder(goal_info_embed)

    @staticmethod
    def _repeat_modes_by_goal(tensor, goal_count):
        if goal_count == 1:
            return tensor
        bs, num_modes = tensor.shape[:2]
        return tensor[:, None].expand(-1, goal_count, *([-1] * (tensor.ndim - 1))).reshape(
            bs,
            goal_count * num_modes,
            *tensor.shape[2:],
        )

    @staticmethod
    def _add_goal_feature(traj_feature, goal_feature, modes_per_goal):
        bs, goal_count, d_model = goal_feature.shape
        query_goal_feature = goal_feature[:, :, None, :].expand(-1, -1, modes_per_goal, -1)
        query_goal_feature = query_goal_feature.reshape(bs, goal_count * modes_per_goal, d_model)
        return traj_feature + query_goal_feature

    @staticmethod
    def _build_per_query_goal_inputs(goalpoint, points_score, modes_per_goal):
        bs, goal_count, _ = goalpoint.shape
        query_goalpoint = goalpoint[:, :, None, :].expand(-1, -1, modes_per_goal, -1)
        query_goalpoint = query_goalpoint.reshape(bs, goal_count * modes_per_goal, 1, 2)
        query_points_score = points_score[:, :, None, :].expand(-1, -1, modes_per_goal, -1)
        query_points_score = query_points_score.reshape(bs, goal_count * modes_per_goal, 1, 2)
        return query_goalpoint, query_points_score

    @staticmethod
    def _build_global_mode_targets(goalpoint, targets, plan_anchor):
        target_xy = targets["trajectory"][..., :2]
        goal_dist = torch.linalg.norm(goalpoint - target_xy[:, -1:, :], dim=-1)
        goal_idx = torch.argmin(goal_dist, dim=-1)

        anchor_dist = torch.linalg.norm(target_xy.unsqueeze(1) - plan_anchor, dim=-1)
        anchor_dist = anchor_dist.mean(dim=-1)
        anchor_idx = torch.argmin(anchor_dist, dim=-1)
        return goal_idx * plan_anchor.shape[1] + anchor_idx

    def _build_goal_uncertainty_negative_targets(self, targets, goalpoint, points_score):
        if targets is None:
            return targets, None
        if not getattr(self._config, "use_beyonddrive", False):
            return targets, None
        if not getattr(self._config, "use_goal_unc_negative_coupling", False):
            return targets, None
        if "negative_trajectory" not in targets:
            return targets, None
        if goalpoint is None or points_score is None:
            return targets, None

        coupling_weight = float(getattr(self._config, "goal_unc_negative_coupling_weight", 0.0))
        if coupling_weight <= 0:
            return targets, None

        negative_trajectory = targets["negative_trajectory"].to(goalpoint)
        negative_index = negative_trajectory.sum(-1).sum(-1) > 0
        if not negative_index.any():
            return targets, None

        negative_endpoint = negative_trajectory[:, -1, :2]
        goal_distance = torch.linalg.norm(goalpoint - negative_endpoint[:, None], dim=-1)
        negative_goal_idx = torch.argmin(goal_distance, dim=-1)

        goal_uncertainty = points_score.abs().mean(dim=-1)
        selected_uncertainty = torch.gather(
            goal_uncertainty,
            dim=1,
            index=negative_goal_idx[:, None],
        ).squeeze(1)

        if goal_uncertainty.shape[1] > 1:
            uncertainty_center = goal_uncertainty.mean(dim=1)
            uncertainty_scale = goal_uncertainty.std(dim=1, unbiased=False).clamp_min(1e-4)
            relative_uncertainty = ((selected_uncertainty - uncertainty_center) / uncertainty_scale).clamp(-1.0, 1.0)
        else:
            relative_uncertainty = torch.zeros_like(selected_uncertainty)

        negative_weight = 1.0 + coupling_weight * relative_uncertainty
        negative_weight = negative_weight.clamp(1.0 - coupling_weight, 1.0 + coupling_weight)
        negative_weight = torch.where(negative_index, negative_weight, torch.ones_like(negative_weight))

        loss_targets = dict(targets)
        loss_targets["negative_goal_uncertainty_weight"] = negative_weight.detach()
        return loss_targets, negative_weight

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
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,goalpoint=None,points_score=1.0) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,goalpoint=goalpoint,points_score=points_score)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=goalpoint,points_score=points_score)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,goalpoint=None,points_score=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        goal_info_feature = self._encode_goal_condition(goalpoint, points_score)
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        modes_per_goal = noisy_traj_points.shape[1]
        noisy_traj_points = self._repeat_modes_by_goal(noisy_traj_points, goal_count)
        plan_anchor_for_loss = self._repeat_modes_by_goal(plan_anchor, goal_count)
        global_mode_idx = None
        cls_avg_factor = None
        if goal_count > 1:
            global_mode_idx = self._build_global_mode_targets(goalpoint, targets, plan_anchor)
            cls_avg_factor = bs * modes_per_goal
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(
            goalpoint,
            points_score,
            modes_per_goal,
        )
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
        loss_targets, negative_goal_uncertainty_weight = self._build_goal_uncertainty_negative_targets(
            targets,
            goalpoint,
            points_score,
        )
        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = self._add_goal_feature(
            traj_feature.view(bs,ego_fut_mode,-1),
            goal_info_feature,
            modes_per_goal,
        )
        # traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)

        navi_points=query_goalpoint
        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,decoder_points_score)

        trajectory_loss_dict = {}
        if negative_goal_uncertainty_weight is not None:
            trajectory_loss_dict["negative_goal_uncertainty_weight"] = negative_goal_uncertainty_weight.mean().detach()
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, loss_targets, plan_anchor_for_loss, mode_idx=global_mode_idx, cls_avg_factor=cls_avg_factor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {
            "trajectory": best_reg,
            "candidate_trajectories": poses_reg_list[-1],
            "candidate_logits": poses_cls_list[-1],
            "trajectory_loss": ret_traj_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
        }

    def sample_chain(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        goalpoint=None,
        points_score=1.0,
        num_samples_per_anchor: Optional[int] = None,
        deterministic: bool = False,
        sampling_denoising_std: Optional[float] = None,
        randn_clip_value: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        step_num = int(getattr(self._config, "grpo_step_num", 2))
        bs = ego_query.shape[0]
        anchor = self.plan_anchor.shape[0]
        num_samples = 64 if num_samples_per_anchor is None else int(num_samples_per_anchor)
        device = ego_query.device
        if sampling_denoising_std is None:
            sampling_denoising_std = float(getattr(self._config, "grpo_min_sampling_denoising_std", 0.04))
        if randn_clip_value is None:
            randn_clip_value = getattr(self._config, "grpo_randn_clip_value", 5.0)
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        goal_info_feature = self._encode_goal_condition(goalpoint, points_score)
        self.diffusion_scheduler.set_timesteps(1000, device)
        timestep_span = int(getattr(self._config, "grpo_timestep_span", 20))
        step_ratio = timestep_span / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        plan_anchor = plan_anchor.unsqueeze(2).repeat(1,1,num_samples,1,1)
        plan_anchor = plan_anchor.view(bs,num_samples*anchor, 8, 2)
        modes_per_goal = plan_anchor.shape[1]
        plan_anchor = self._repeat_modes_by_goal(plan_anchor, goal_count)
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(
            goalpoint,
            points_score,
            modes_per_goal,
        )
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        if randn_clip_value is not None:
            noise = noise.clamp(-float(randn_clip_value), float(randn_clip_value))
        trunc_timestep = int(getattr(self._config, "grpo_trunc_timestep", 8))
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * trunc_timestep
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        chain = [img.clone()]
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_feature(
                traj_feature.view(bs,ego_fut_mode,-1),
                goal_info_feature,
                modes_per_goal,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            navi_points = query_goalpoint
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,decoder_points_score)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            mean = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
            if deterministic:
                img = mean
            else:
                noise = torch.randn_like(mean)
                if randn_clip_value is not None:
                    noise = noise.clamp(-float(randn_clip_value), float(randn_clip_value))
                img = mean + float(sampling_denoising_std) * noise
            chain.append(img.clone())
        selected_candidate_index = poses_cls.argmax(dim=-1)
        mode_idx = selected_candidate_index[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        candidate_trajectories = poses_reg
        candidate_logits = poses_cls
        if goal_count == 1:
            anchor_trajectories = poses_reg.view(bs, anchor, num_samples, self._num_poses, 3)
        else:
            anchor_trajectories = poses_reg.view(bs, goal_count, anchor, num_samples, self._num_poses, 3)

        return {
            "chain": torch.stack(chain, dim=1).detach(),
            "trajectory": best_reg.detach(),
            "candidate_trajectories": candidate_trajectories.detach(),
            "candidate_logits": candidate_logits.detach(),
            "selected_candidate_index": selected_candidate_index.detach(),
            "anchor_trajectories": anchor_trajectories.detach(),
        }

    def get_logprobs(
        self,
        chains,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        global_img,
        goalpoint=None,
        points_score=1.0,
        selected_candidate_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = ego_query.shape[0]
        num_modes = chains.shape[2]
        anchor = self.plan_anchor.shape[0]
        device = chains.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        modes_per_goal = max(1, num_modes // goal_count)
        goal_info_feature = self._encode_goal_condition(goalpoint, points_score)
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(
            goalpoint,
            points_score,
            modes_per_goal,
        )
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
        step_num = int(getattr(self._config, "grpo_step_num", chains.shape[1] - 1))
        timestep_span = int(getattr(self._config, "grpo_timestep_span", 20))
        step_ratio = timestep_span / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        log_probs = []
        for step_idx, k in enumerate(roll_timesteps[:chains.shape[1] - 1]):
            img = chains[:, step_idx]
            next_img = chains[:, step_idx + 1]
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_feature(
                traj_feature.view(bs,num_modes,-1),
                goal_info_feature,
                modes_per_goal,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)

            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            poses_reg_list, _ = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,query_goalpoint,decoder_points_score)
            x_start = poses_reg_list[-1][...,:2]
            x_start = self.norm_odo(x_start)
            mean = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
            logprob_target = next_img
            if selected_candidate_index is not None:
                selected_index = selected_candidate_index.to(device=mean.device, dtype=torch.long)
                if selected_index.ndim != 1:
                    selected_index = selected_index.view(selected_index.shape[0])
                selected_index = selected_index.clamp(0, mean.shape[1] - 1)
                gather_index = selected_index[:, None, None, None].expand(
                    -1,
                    1,
                    mean.shape[2],
                    mean.shape[3],
                )
                mean = torch.gather(mean, dim=1, index=gather_index).squeeze(1)
                logprob_target = torch.gather(logprob_target, dim=1, index=gather_index).squeeze(1)
            std = torch.full_like(mean, float(getattr(self._config, "grpo_min_logprob_denoising_std", 0.1)))
            log_probs.append(Normal(mean, std).log_prob(logprob_target).mean(dim=list(range(1, mean.ndim))))

        return torch.stack(log_probs, dim=1)

    @staticmethod
    def _load_metric_cache(metric_cache_path):
        with lzma.open(metric_cache_path, "rb") as f:
            return pickle.load(f)

    def reward_fn(self, pred_traj: torch.Tensor, metric_cache_paths) -> torch.Tensor:
        pred_np = pred_traj.detach().float().cpu().numpy()
        rewards = []
        for i, metric_cache_path in enumerate(metric_cache_paths):
            metric_cache = self._load_metric_cache(metric_cache_path)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(pred_np[i], trajectory_sampling=self._config.trajectory_sampling),
                future_sampling=self._config.trajectory_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )
            rewards.append(asdict(pdm_result)["score"])
        return torch.tensor(rewards, device=pred_traj.device, dtype=torch.float32).detach()

    def forward_grpo(
        self,
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=None,
        global_img=None,
        goalpoint=None,
        points_score=None,
        metric_cache_paths=None,
        wm_selector=None,
        wm_keyval=None,
    ) -> Dict[str, torch.Tensor]:
        if metric_cache_paths is None:
            raise ValueError("metric_cache_paths are required for Mimir GRPO training.")
        if not hasattr(self, "simulator"):
            self.simulator = PDMSimulator(self._config.trajectory_sampling)
            self.train_scorer = PDMScorer(self._config.trajectory_sampling, self._config.grpo_scorer_config)

        diff_decoder_was_training = self.diff_decoder.training
        self.diff_decoder.eval()

        batch_size = ego_query.shape[0]
        sample_time = int(getattr(self._config, "grpo_sample_time", 4))
        ego_query_rep = ego_query.repeat_interleave(sample_time, 0)
        agents_query_rep = agents_query.repeat_interleave(sample_time, 0)
        bev_feature_rep = bev_feature.repeat_interleave(sample_time, 0)
        status_encoding_rep = status_encoding.repeat_interleave(sample_time, 0)
        metric_cache_paths_rep = [path for path in metric_cache_paths for _ in range(sample_time)]
        if global_img is None:
            global_img_rep = None
        else:
            global_img_rep = global_img.repeat_interleave(sample_time, 0)
        if goalpoint is None:
            goalpoint_rep = None
        else:
            goalpoint_rep = goalpoint.repeat_interleave(sample_time, 0)
        if points_score is None:
            points_score_rep = None
        else:
            points_score_rep = points_score.repeat_interleave(sample_time, 0)

        with torch.no_grad():
            sampled = self.sample_chain(
                ego_query_rep,
                agents_query_rep,
                bev_feature_rep,
                bev_spatial_shape,
                status_encoding_rep,
                global_img_rep,
                goalpoint=goalpoint_rep,
                points_score=points_score_rep,
                num_samples_per_anchor=int(getattr(self._config, "grpo_num_samples_per_anchor", 1)),
            )
            if wm_selector is not None and wm_keyval is not None:
                sampled = wm_selector(sampled, wm_keyval)
        rewards = self.reward_fn(sampled["trajectory"], metric_cache_paths_rep).float()
        rewards_matrix = rewards.view(batch_size, sample_time)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
        advantages = ((rewards_matrix - mean_r) / std_r).reshape(-1).detach()
        adv_min = torch.quantile(advantages, float(getattr(self._config, "grpo_clip_advantage_lower_quantile", 0.0)))
        adv_max = torch.quantile(advantages, float(getattr(self._config, "grpo_clip_advantage_upper_quantile", 1.0)))
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        log_probs = self.get_logprobs(
            sampled["chain"],
            ego_query_rep,
            agents_query_rep,
            bev_feature_rep,
            bev_spatial_shape,
            status_encoding_rep,
            global_img_rep,
            goalpoint=goalpoint_rep,
            points_score=points_score_rep,
            selected_candidate_index=sampled.get("selected_candidate_index"),
        )
        log_probs = log_probs.clamp(
            min=float(getattr(self._config, "grpo_logprob_clip_min", -5.0)),
            max=float(getattr(self._config, "grpo_logprob_clip_max", 2.0)),
        )
        num_steps = log_probs.shape[1]
        denoising_indices = torch.arange(num_steps, device=log_probs.device, dtype=log_probs.dtype)
        discount = float(getattr(self._config, "grpo_gamma_denoising", 0.6)) ** (num_steps - denoising_indices - 1)
        policy_loss = -(log_probs * advantages[:, None] * discount[None]).mean()
        total_loss = policy_loss

        bc_loss = log_probs.new_tensor(0.0)
        if getattr(self._config, "grpo_use_bc_loss", True) and self.old_policy is not None:
            with torch.no_grad():
                teacher = self.old_policy.sample_chain(
                    ego_query,
                    agents_query,
                    bev_feature,
                    bev_spatial_shape,
                    status_encoding,
                    global_img,
                    goalpoint=goalpoint,
                    points_score=points_score,
                    num_samples_per_anchor=int(getattr(self._config, "grpo_num_samples_per_anchor", 1)),
                )
                if wm_selector is not None and wm_keyval is not None:
                    teacher = wm_selector(teacher, wm_keyval)
            bc_logp = self.get_logprobs(
                teacher["chain"],
                ego_query,
                agents_query,
                bev_feature,
                bev_spatial_shape,
                status_encoding,
                global_img,
                goalpoint=goalpoint,
                points_score=points_score,
                selected_candidate_index=teacher.get("selected_candidate_index"),
            )
            bc_logp = bc_logp.clamp(
                min=float(getattr(self._config, "grpo_logprob_clip_min", -5.0)),
                max=float(getattr(self._config, "grpo_logprob_clip_max", 2.0)),
            )
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + float(getattr(self._config, "grpo_bc_coeff", 0.1)) * bc_loss

        self.diff_decoder.train(diff_decoder_was_training)

        return {
            "trajectory": sampled["trajectory"].view(batch_size, sample_time, self._num_poses, 3)[:, 0],
            "loss": total_loss,
            "reward": rewards.mean(),
            "policy_loss": policy_loss,
            "bc_loss": bc_loss,
        }

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=None,points_score=1.0) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        anchor = 20
        num_samples = 64
        device = ego_query.device
        goalpoint, points_score = self._prepare_goal_inputs(goalpoint, points_score, bs, device, ego_query.dtype)
        goal_count = goalpoint.shape[1]
        goal_info_feature = self._encode_goal_condition(goalpoint, points_score)
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        plan_anchor = plan_anchor.unsqueeze(2).repeat(1,1,num_samples,1,1)
        plan_anchor = plan_anchor.view(bs,num_samples*anchor, 8, 2)
        modes_per_goal = plan_anchor.shape[1]
        plan_anchor = self._repeat_modes_by_goal(plan_anchor, goal_count)
        query_goalpoint, query_points_score = self._build_per_query_goal_inputs(
            goalpoint,
            points_score,
            modes_per_goal,
        )
        decoder_points_score = query_points_score if self._config.use_unc_score else 1.0
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = self._add_goal_feature(
                traj_feature.view(bs,ego_fut_mode,-1),
                goal_info_feature,
                modes_per_goal,
            )

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            navi_points = query_goalpoint
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,decoder_points_score)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        candidate_trajectories = poses_reg
        candidate_logits = poses_cls
        if goal_count == 1:
            poses_reg = poses_reg.view(bs, anchor, num_samples, self._num_poses, 3)
        else:
            poses_reg = poses_reg.view(bs, goal_count, anchor, num_samples, self._num_poses, 3)
                
        return {"trajectory": best_reg,
                "candidate_trajectories": candidate_trajectories,
                "candidate_logits": candidate_logits,
                'anchor_trajectories': poses_reg
                }
