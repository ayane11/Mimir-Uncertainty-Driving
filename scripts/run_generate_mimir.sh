# ======================================================= navtest ==============================================================================
TRAIN_TEST_SPLIT=navtrain
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=5

export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
export OPENBLAS_CORETYPE=Haswell

CHECKPOINT_PATH='/home/navsim/ckpt/mimir_unc_epoch99.ckpt'
CACHE_PATH=/home/navsim/exp/cache/navtrain_metric_cache
GOAL_COORD_PATH='/home/navsim/dataset/goal/navtrain_top3_1m.npy'


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_unc_navtest.py \
agent=mimir_agent_unc \
experiment_name=a_navtest_mimir_agent_unc_eval \
train_test_split=$TRAIN_TEST_SPLIT \
agent.config.latent=False \
agent.config.training=False \
agent.config.use_proj_image=False \
agent.config.use_gt_goal_train=False \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.status_norm=False \
agent.config.num_goal_points=3 \
agent.config.navi_bank_path=$GOAL_COORD_PATH \
agent.config.navi_unc_outputdir='/home/navsim/dataset/navtrain_3goals_naviunc/' \
agent.config.goal_coord_path=$GOAL_COORD_PATH \
metric_cache_path=$CACHE_PATH \
worker.threads_per_node=32