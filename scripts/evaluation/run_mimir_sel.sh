export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
export OPENBLAS_CORETYPE=Haswell

# ====================================== use unc to train ==============================================================
METRIC_CACHE_PATH='/home/navsim/exp/metric_cache'
CHECKPOINT_PATH='/home/navsim/exp/a_train_mimir_agent_sel/2026.08.05.15.19.59/lightning_logs/version_0/checkpoints/best-epoch\=1.ckpt'
GOAL_COORD_PATH='/home/navsim/dataset/naviunc/a_navtest_3_05_merge/navi_dict.npy'
UNC_PATH='/home/navsim/dataset/naviunc/a_navtest_3_05_merge/unc_dict.npy'
TRAJ_SAVE_PATH='/home/navsim/dataset/trajs'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    agent=mimir_sel_agent \
    train_test_split=navtest \
    experiment_name=a_navtest_mimir_agent_traj_eval \
    worker=ray_distributed \
    metric_cache_path=$METRIC_CACHE_PATH \
    +agent.traj_save_path=$TRAJ_SAVE_PATH \
    agent.config.latent=False \
    agent.config.training=False \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.checkpoint_path=$CHECKPOINT_PATH \
    agent.config.status_norm=False \
    agent.config.use_unc_score=True \
    agent.config.navi_path=$GOAL_COORD_PATH \
    agent.config.unc_path=$UNC_PATH \
    agent.config.use_wm=True \
    agent.config.fusion_base_weight=1.0 \
    agent.config.fusion_wm_weight=0.5 \
    agent.config.weight=1.0 \
    agent.config.num_samples=64 \
    agent.config.num_goal_points=3