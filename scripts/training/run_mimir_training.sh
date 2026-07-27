TRAIN_TEST_SPLIT=navtrain
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
COORD_PATH='/home/navsim/dataset/naviunc/a_navtrain_3_05_merge/navi_dict.npy'
UNC_PATH='/home/navsim/dataset/naviunc/a_navtrain_3_05_merge/unc_dict.npy'
NEGATIVE_SAMPLES_PATH='/home/navsim/dataset/sample_step5_cfg0.5_noise0.4_proposal64/2026.01.07.03.26.47/negative_sample_results/'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=mimir_final_agent \
    experiment_name=a_navtrain_mimir_agent_traj \
    train_test_split=$TRAIN_TEST_SPLIT \
    use_cache_without_dataset=True \
    cache_path=$CACHE_PATH \
    negative_samples_path=$NEGATIVE_SAMPLES_PATH \
    dataloader.params.batch_size=64 \
    trainer.params.max_epochs=100 \
    force_cache_computation=False \
    agent.config.latent=False \
    agent.config.training=True \
    agent.config.num_goal_points=3 \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.config.status_norm=False \
    agent.config.use_unc_score=True \
    agent.config.use_beyonddrive=True \
    agent.config.unc_path=$UNC_PATH \
    agent.config.navi_path=$COORD_PATH \
    agent.config.use_wm=True