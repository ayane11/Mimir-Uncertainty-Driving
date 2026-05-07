export OPENBLAS_CORETYPE=Haswell
export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/home/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/home/navsim/Mimir-Uncertainty-Driving"
export OPENSCENE_DATA_ROOT="/home/navsim/dataset"
CACHE_TO_SAVE='/data/openscene/cache/mimir_feature_cache' #set your feature cache path to save
export HYDRA_FULL_ERROR=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=mimir_wm_agent \
experiment_name=a_mimir_trainval_feature_cache \
cache_path=$CACHE_TO_SAVE \
train_test_split=navtrain \
agent.config.latent=False \
worker.threads_per_node=32