export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
export OPENBLAS_CORETYPE=Haswell

# ====================================== use unc to train ==============================================================
CHECKPOINT_PATH='/home/navsim/ckpt/mimir_epoch94.ckpt'
GOAL_COORD_PATH='/home/navsim/dataset/navtest_naviunc/navi.npy'
UNC_PATH='/home/navsim/dataset/navtest_naviunc/unc.npy'
TRAJ_SAVE_PATH='/home/navsim/dataset/trajs'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    agent=mimir_agent \
    train_test_split=navtest \
    experiment_name=a_navtest_mimir_agent_traj_eval \
    worker=ray_distributed \
    agent.traj_save_path=$TRAJ_SAVE_PATH \
    agent.config.latent=False \
    agent.config.training=False \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.checkpoint_path=$CHECKPOINT_PATH \
    agent.config.status_norm=False \
    agent.config.use_unc_score=True \
    agent.config.navi_path=$GOAL_COORD_PATH \
    agent.config.unc_path=$UNC_PATH