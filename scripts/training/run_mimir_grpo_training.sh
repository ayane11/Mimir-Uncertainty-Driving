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

CACHE_PATH='/data/openscene/cache/mimir_feature_cache'
TRAIN_METRIC_CACHE_PATH='/data/openscene/cache/navtrain_metric_cache'
CHECKPOINT_PATH='/home/navsim/exp/a_navtrain_mimir_agent_traj/mimir_final_90.0/lightning_logs/version_0/checkpoints/best-epoch\=96.ckpt'
COORD_PATH='/home/navsim/dataset/naviunc/a_navtrain_3_05_merge/navi_dict.npy'
UNC_PATH='/home/navsim/dataset/naviunc/a_navtrain_3_05_merge/unc_dict.npy'

# GRPO reward requires PDM metric caches under:
# $TRAIN_METRIC_CACHE_PATH/<log>/$TRAIN_METRIC_CACHE_SCENARIO_TYPE/<token>/metric_cache.pkl
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=mimir_grpo_agent \
    experiment_name=a_train_mimir_agent_grpo \
    train_test_split=$TRAIN_TEST_SPLIT \
    split=trainval \
    use_cache_without_dataset=True \
    cache_path=$CACHE_PATH \
    +train_metric_cache_path=$TRAIN_METRIC_CACHE_PATH \
    dataloader.params.batch_size=32 \
    trainer.params.max_epochs=10 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    force_cache_computation=False \
    agent.checkpoint_path=$CHECKPOINT_PATH \
    agent.config.latent=False \
    agent.config.training=True \
    agent.config.use_proj_image=False \
    agent.config.use_gt_goal_train=False \
    agent.config.status_norm=False \
    agent.config.use_unc_score=True \
    agent.config.use_wm=True \
    agent.config.grpo=True \
    agent.config.unc_path=$UNC_PATH \
    agent.config.navi_path=$COORD_PATH \
    agent.lr=5e-6 \
    agent.config.grpo_sample_time=8 \
    agent.config.grpo_num_samples_per_anchor=1 \
    +agent.config.grpo_step_num=2 \
    +agent.config.grpo_timestep_span=20 \
    +agent.config.grpo_trunc_timestep=8 \
    agent.config.grpo_bc_coeff=0.2 \
    +agent.config.grpo_use_bc_loss=True \
    +agent.config.grpo_gamma_denoising=0.6 \
    +agent.config.grpo_min_sampling_denoising_std=0.04 \
    +agent.config.grpo_min_logprob_denoising_std=0.1 \
    +agent.config.grpo_randn_clip_value=5.0 \
    +agent.config.grpo_logprob_clip_min=-5.0 \
    +agent.config.grpo_logprob_clip_max=2.0 \
    +agent.config.grpo_clip_advantage_lower_quantile=0.0 \
    +agent.config.grpo_clip_advantage_upper_quantile=1.0 \