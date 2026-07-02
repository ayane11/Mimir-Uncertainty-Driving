import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_configured_bev_on_ax, add_navi_bank_to_bev_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


def _parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_output_stem(output_name: Optional[str], token: str) -> str:
    if output_name is None or output_name == "":
        return f"{token}_trajectory"
    return output_name.format(token=token)


def _resolve_trajectory_path(traj_path: Path, token: str) -> Path:
    if traj_path.is_dir() or traj_path.suffix != ".npy":
        return traj_path / f"{token}.npy"
    return traj_path


def _load_saved_trajectory(
    trajectory_path: Path,
    interval_length: float,
) -> Trajectory:
    poses = np.asarray(np.load(trajectory_path), dtype=np.float32)
    poses = np.squeeze(poses)

    if poses.ndim != 2:
        raise ValueError(f"Expected saved trajectory with shape (T, 2/3), got {poses.shape} from {trajectory_path}.")
    if poses.shape[-1] < 2:
        raise ValueError(f"Trajectory must contain at least x/y coordinates, got {poses.shape} from {trajectory_path}.")
    if poses.shape[-1] == 2:
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 1), dtype=poses.dtype)], axis=-1)
    if poses.shape[-1] > 3:
        poses = poses[:, :3]

    trajectory_sampling = TrajectorySampling(
        time_horizon=poses.shape[0] * interval_length,
        interval_length=interval_length,
    )
    return Trajectory(poses, trajectory_sampling)


def _agent_plot_config() -> Dict:
    config = dict(TRAJECTORY_CONFIG["agent"])
    config.update(
        {
            "line_color": "#1f77b4",
            "line_width": 2.8,
            "marker_size": 4,
            "zorder": 5,
        }
    )
    return config


def _draw_plan(
    scene,
    token: str,
    prediction: Trajectory,
    output_path: Path,
    navi_bank: Optional[Path],
    unc_bank: Optional[Path],
    num_goal_points: int,
    show_gt: bool,
    show_goals: bool,
    show_colorbar: bool,
    dpi: int,
) -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])

    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    if show_gt:
        add_trajectory_to_bev_ax(ax, scene.get_future_trajectory(prediction.poses.shape[0]), TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, prediction, _agent_plot_config())
    if show_goals and navi_bank is not None:
        add_navi_bank_to_bev_ax(
            ax,
            navi_bank=navi_bank,
            token=token,
            unc_bank=unc_bank,
            num_goal_points=num_goal_points,
            show_colorbar=show_colorbar,
            labels=True,
        )

    configure_bev_ax(ax)
    configure_ax(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a saved NAVSIM trajectory on one scene.")
    parser.add_argument("--token", required=True, type=str, help="NAVSIM token to visualize.")
    parser.add_argument(
        "--traj_path",
        required=True,
        type=Path,
        help="Saved trajectory npy file, or a directory containing {token}.npy.",
    )
    parser.add_argument("--navsim_log_path", required=True, type=Path, help="Path to navsim_logs split directory.")
    parser.add_argument("--sensor_blobs_path", required=True, type=Path, help="Path to sensor_blobs split directory.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to save rendered figure.")
    parser.add_argument(
        "--output_name",
        default=None,
        type=str,
        help="Optional output file stem without extension. Supports {token}, e.g. mimir_final_{token}.",
    )
    parser.add_argument("--log_names", default=None, type=str, help="Optional comma-separated log names for faster loading.")
    parser.add_argument("--navi_bank", default=None, type=Path, help="Optional token-indexed navi_dict.npy.")
    parser.add_argument("--unc_bank", default=None, type=Path, help="Optional token-indexed unc_dict.npy.")
    parser.add_argument("--num_goal_points", default=3, type=int)
    parser.add_argument("--num_history_frames", default=4, type=int)
    parser.add_argument("--num_future_frames", default=10, type=int)
    parser.add_argument("--frame_interval", default=1, type=int)
    parser.add_argument("--interval_length", default=0.5, type=float)
    parser.add_argument("--no_gt", action="store_true", help="Do not draw expert/GT future trajectory.")
    parser.add_argument("--no_goals", action="store_true", help="Do not draw navi goals.")
    parser.add_argument("--no_colorbar", action="store_true", help="Do not draw goal uncertainty colorbar.")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", default=300, type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = _resolve_trajectory_path(args.traj_path, args.token)
    if not trajectory_path.exists():
        raise RuntimeError(f"Saved trajectory was not found: {trajectory_path}")
    prediction = _load_saved_trajectory(trajectory_path, args.interval_length)
    print(f"Loaded trajectory {trajectory_path}")

    scene_filter = SceneFilter(
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        has_route=True,
        log_names=_parse_csv_arg(args.log_names),
        tokens=[args.token],
    )
    scene_loader = SceneLoader(
        data_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[args.num_history_frames - 1]),
    )
    if args.token not in scene_loader.tokens:
        raise RuntimeError(f"Token {args.token} was not found under {args.navsim_log_path}.")

    scene = scene_loader.get_scene_from_token(args.token)
    output_stem = _format_output_stem(args.output_name, args.token)
    output_path = args.output_dir / f"{output_stem}.{args.format}"
    _draw_plan(
        scene=scene,
        token=args.token,
        prediction=prediction,
        output_path=output_path,
        navi_bank=args.navi_bank,
        unc_bank=args.unc_bank,
        num_goal_points=args.num_goal_points,
        show_gt=not args.no_gt,
        show_goals=not args.no_goals,
        show_colorbar=not args.no_colorbar,
        dpi=args.dpi,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
