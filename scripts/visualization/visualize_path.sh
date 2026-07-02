export PYTHONPATH=/home/navsim/Mimir-Uncertainty-Driving/
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/home/navsim/dataset/maps
export OPENSCENE_DATA_ROOT=/home/navsim/dataset
export OPENBLAS_CORETYPE=Haswell

TOKEN=5d7c7edca69d5e73

GOAL_COORD_PATH='/home/navsim/dataset/naviunc/a_navtest_3_05_merge/navi_dict.npy'
UNC_PATH='/home/navsim/dataset/naviunc/a_navtest_3_05_merge/unc_dict.npy'
traj_path='/home/navsim/dataset/trajs/mimir_final/'

python3 scripts/visualization/visualize_path.py \
  --token $TOKEN \
  --traj_path $traj_path \
  --navsim_log_path /home/navsim/dataset/navsim_logs/test \
  --sensor_blobs_path /home/navsim/dataset/sensor_blobs/test \
  --navi_bank $GOAL_COORD_PATH \
  --unc_bank $UNC_PATH \
  --output_dir /home/navsim/figures/$TOKEN/ \
  --output_name mimir_final_{token} \
  --num_goal_points 3 \
  --no_gt \
  --no_goals \
  --format png

GOAL_COORD_PATH=/home/navsim/dataset/naviunc/navtest_naviunc/navi_dict.npy
UNC_PATH=/home/navsim/dataset/naviunc/navtest_naviunc/unc_dict.npy
traj_path='/home/navsim/dataset/trajs/mimir/'

python3 scripts/visualization/visualize_path.py \
  --token $TOKEN \
  --traj_path $traj_path \
  --navsim_log_path /home/navsim/dataset/navsim_logs/test \
  --sensor_blobs_path /home/navsim/dataset/sensor_blobs/test \
  --navi_bank $GOAL_COORD_PATH \
  --unc_bank $UNC_PATH \
  --output_dir /home/navsim/figures/$TOKEN/ \
  --output_name mimir_{token} \
  --num_goal_points 1 \
  --format png \
  --no_gt \
  --no_goals

traj_path='/home/navsim/dataset/trajs/diffusiondrive/'

python3 scripts/visualization/visualize_path.py \
  --token $TOKEN \
  --traj_path $traj_path \
  --navsim_log_path /home/navsim/dataset/navsim_logs/test \
  --sensor_blobs_path /home/navsim/dataset/sensor_blobs/test \
  --output_dir /home/navsim/figures/$TOKEN/ \
  --output_name diffusiondrive_{token} \
  --format png \
  --no_gt \
  --no_goals

