###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/home/mc/wangjiahao/nuplan/dataset/nuplan-v1.1/splits/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/home/mc/wangjiahao/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/home/mc/wangjiahao/project/Diffusion-Planner/train_set" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--use_multiprocessing \
--num_workers 8 \
--total_scenarios 1000000 \

