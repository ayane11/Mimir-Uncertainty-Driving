TRAIN_TEST_SPLIT=navtrain
export CUDA_VISIBLE_DEVICES="0"
export HYDRA_FULL_ERROR=1

export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
export OPENBLAS_CORETYPE=Haswell

# ============================================ training for unc ====================================================
CACHE_PATH='/data/openscene/cache/mimir_feature_cache'
TRAIN_METRIC_CACHE_PATH='/data/openscene/cache/navtrain_metric_cache'
CHECKPOINT_PATH='/home/navsim/exp/a_train_mimir_agent_rl/2026.05.13.19.19.36/lightning_logs/version_0/checkpoints/epoch\=9-step\=13300.ckpt'
COORD_PATH='/home/navsim/dataset/naviunc/navtrain_full_naviunc/navi_dict.npy'
UNC_PATH='/home/navsim/dataset/naviunc/navtrain_full_naviunc/unc_dict.npy'

# Selector reward labels require PDM metric caches under:
# $TRAIN_METRIC_CACHE_PATH/<log>/$TRAIN_METRIC_CACHE_SCENARIO_TYPE/<token>/metric_cache.pkl
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=mimir_sel_agent \
    experiment_name=a_train_mimir_agent_sel \
    train_test_split=$TRAIN_TEST_SPLIT \
    split=trainval \
    use_cache_without_dataset=True \
    cache_path=$CACHE_PATH \
    +train_metric_cache_path=$TRAIN_METRIC_CACHE_PATH \
    dataloader.params.batch_size=64 \
    trainer.params.max_epochs=20 \
    force_cache_computation=False \
    agent.checkpoint_path=$CHECKPOINT_PATH \
    agent.config.latent=False \
    agent.config.training=True \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.config.status_norm=False \
    agent.config.use_unc_score=True \
    agent.config.use_wm=True \
    agent.config.unc_path=$UNC_PATH \
    agent.config.navi_path=$COORD_PATH