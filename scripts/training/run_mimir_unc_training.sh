TRAIN_TEST_SPLIT=navtrain
export CUDA_VISIBLE_DEVICES="3"

export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
export OPENBLAS_CORETYPE=Haswell

CACHE_PATH='/home/navsim/exp/cache/mimir_feature_cache'
COORD_PATH='/home/navsim/dataset/goal/navtrain_top3_1m.npy'


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=mimir_agent_unc \
    experiment_name=a_navtrain_mimir_agent_unc \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    cache_path=$CACHE_PATH \
    dataloader.params.batch_size=64 \
    trainer.params.max_epochs=100 \
    force_cache_computation=False \
    agent.config.navi_bank_path=$COORD_PATH \
    agent.config.latent=False \
    agent.config.training=True \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.config.status_norm=False \
    agent.lr=1.5e-4