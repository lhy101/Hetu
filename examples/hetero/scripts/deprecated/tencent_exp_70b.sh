#!/bin/bash

# bash scripts/hot_switch_multi.sh 7b two bf16 greedy host01

MODEL_SIZE=${1:-'70b'}
SWITCH=${2:-0}
HOSTFILE=${3:-'hostfile01234567'}
SEQ_LEN=${4:-4096}
GLOBAL_BATCH_SIZE=${5:-64}
MICRO_BATCH_SIZE=${6:-1}
DP=${7:-2}
TP=${7:-8}
PP=${7:-4}
HETERO=true

if [ "${MODEL_SIZE}" = "7b" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    FFN_HIDDEN_SIZE=13824
    NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "30b" ]; then
    # actually 30b = 12*num_layers*hidden_size^2
    NUM_LAYERS=60
    HIDDEN_SIZE=6528 #6672
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=48 # should be divided by tp32... so 48 will cause error!!!
elif [ "${MODEL_SIZE}" = "32b" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656 #6672
    # HIDDEN_SIZE=512
    # FFN_HIDDEN_SIZE=2752
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70b" ]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192 #6672
    FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/13b/30b for test.
    exit 0
fi

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))

# before
BEFORE_LAYERS_NUM_LIST="20,20,20,20,20,20,20,20"
BEFORE_STAGES_NUM_LIST="[4,4]"
BEFORE_MICRO_BATCH_NUM_LIST="[32,32]"
# BEFORE_UNUSED_RANK="[1,2,3,4,5,6,7,9,10,11,12,13,14,15,20,21,22,23,58,59,60,61,62,63,66,67,68,69,70,71,76,77,78,79]"
BEFORE_UNUSED_RANK="[]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31,32:32,33:33,34:34,35:35,36:36,37:37,38:38,39:39,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47,48:48,49:49,50:50,51:51,52:52,53:53,54:54,55:55,56:56,57:57,58:58,59:59,60:60,61:61,62:62,63:63}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:64,3:65,4:66,5:67,6:68,7:69,8:5,9:6,10:70,11:71,12:72,13:73,14:74,15:75,16:1,17:2,18:3,19:4,20:76,21:77,22:78,23:79,24:40,25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:48,33:49,34:50,35:51,36:52,37:53,38:54,39:55,40:56,41:57,42:58,43:59,44:60,45:61,46:62,47:63,48:32,49:33,50:34,51:35,52:36,53:37,54:38,55:39,56:8,57:9,58:10,59:11,60:12,61:13,62:14,63:15,64:16,65:17,66:18,67:19,68:20,69:21,70:22,71:23,72:24,73:25,74:26,75:27,76:28,77:29,78:30,79:31}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:8,3:64,4:65,5:66,6:67,7:68,8:15,9:69,10:70,11:71,12:72,13:73,14:74,15:75,16:1,17:2,18:3,19:4,20:76,21:77,22:78,23:79,24:40,25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:16,33:17,34:18,35:19,36:20,37:21,38:22,39:23,40:24,41:25,42:26,43:27,44:28,45:29,46:30,47:31,48:32,49:33,50:34,51:35,52:36,53:37,54:38,55:39,56:5,57:6,58:80,59:81,60:82,61:83,62:84,63:85,64:13,65:14,66:86,67:87,68:88,69:89,70:90,71:91,72:9,73:10,74:11,75:12,76:92,77:93,78:94,79:95,80:48,81:49,82:50,83:51,84:52,85:53,86:54,87:55,88:56,89:57,90:58,91:59,92:60,93:61,94:62,95:63}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:48,17:49,18:50,19:51,20:52,21:53,22:54,23:55,24:56,25:57,26:58,27:59,28:60,29:61,30:62,31:63,32:40,33:41,34:42,35:43,36:44,37:45,38:46,39:47,40:16,41:17,42:18,43:19,44:20,45:21,46:22,47:23,48:24,49:25,50:26,51:27,52:28,53:29,54:30,55:31,56:32,57:33,58:34,59:35,60:36,61:37,62:38,63:39}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $BEFORE_LAYERS_NUM_LIST \
    --hetero_stages $BEFORE_STAGES_NUM_LIST \
    --rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
    --unused_rank $BEFORE_UNUSED_RANK \
    --file_name "before.json"

# after
AFTER_LAYERS_NUM_LIST="2,5,11,20,21,21,20,20,20,20"
# AFTER_LAYERS_NUM_LIST="3,7,14,28,28,16,16,16,16,16"
AFTER_STAGES_NUM_LIST="[6,4]"
AFTER_MICRO_BATCH_NUM_LIST="[31,33]"
# AFTER_MICRO_BATCH_NUM_LIST="[32,32]"
AFTER_UNUSED_RANK="[1,2,3,4,5,6,7,9,10,11,12,13,14,15,20,21,22,23,58,59,60,61,62,63,66,67,68,69,70,71,76,77,78,79]"
AFTER_UNUSED_RANK="[1,2,3,4,5,6,7,10,11,12,13,14,15,20,21,22,23]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31,32:32,33:33,34:34,35:35,36:36,37:37,38:38,39:39,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47,48:48,49:49,50:50,51:51,52:52,53:53,54:54,55:55,56:56,57:57,58:58,59:59,60:60,61:61,62:62,63:63}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:64,3:65,4:66,5:67,6:68,7:69,8:5,9:6,10:70,11:71,12:72,13:73,14:74,15:75,16:1,17:2,18:3,19:4,20:76,21:77,22:78,23:79,24:40,25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:48,33:49,34:50,35:51,36:52,37:53,38:54,39:55,40:56,41:57,42:58,43:59,44:60,45:61,46:62,47:63,48:32,49:33,50:34,51:35,52:36,53:37,54:38,55:39,56:8,57:9,58:10,59:11,60:12,61:13,62:14,63:15,64:16,65:17,66:18,67:19,68:20,69:21,70:22,71:23,72:24,73:25,74:26,75:27,76:28,77:29,78:30,79:31}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:8,3:64,4:65,5:66,6:67,7:68,8:15,9:69,10:70,11:71,12:72,13:73,14:74,15:75,16:1,17:2,18:3,19:4,20:76,21:77,22:78,23:79,24:40,25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:16,33:17,34:18,35:19,36:20,37:21,38:22,39:23,40:24,41:25,42:26,43:27,44:28,45:29,46:30,47:31,48:32,49:33,50:34,51:35,52:36,53:37,54:38,55:39,56:5,57:6,58:80,59:81,60:82,61:83,62:84,63:85,64:13,65:14,66:86,67:87,68:88,69:89,70:90,71:91,72:9,73:10,74:11,75:12,76:92,77:93,78:94,79:95,80:48,81:49,82:50,83:51,84:52,85:53,86:54,87:55,88:56,89:57,90:58,91:59,92:60,93:61,94:62,95:63}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:48,17:49,18:50,19:51,20:52,21:53,22:54,23:55,24:56,25:57,26:58,27:59,28:60,29:61,30:62,31:63,32:40,33:41,34:42,35:43,36:44,37:45,38:46,39:47,40:16,41:17,42:18,43:19,44:20,45:21,46:22,47:23,48:24,49:25,50:26,51:27,52:28,53:29,54:30,55:31,56:32,57:33,58:34,59:35,60:36,61:37,62:38,63:39}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $AFTER_LAYERS_NUM_LIST \
    --hetero_stages $AFTER_STAGES_NUM_LIST \
    --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
    --unused_rank $AFTER_UNUSED_RANK \
    --file_name "after.json"

python ./ds_parallel_config/generate_gpt_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero 

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export PATH=/jizhicfs/hymiezhao/miniconda3/envs/hetu-py/bin:$PATH
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../" && pwd )"
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${HETU_HOME}/build/hetu/third_party/cutlass/install:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

echo HETU_HOME = $HETU_HOME

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_STRAGGLER=ANALYSIS
export HETU_MEMORY_PROFILE=INFO

export HETU_MAX_SPLIT_SIZE_MB=0
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0

export NCCL_DEBUG=WARN

file="straggler_exp/${MODEL_SIZE}_gpus${WORLD_SIZE}_result.txt"
echo will write result into ${file}...
dir=$(dirname "$file")
if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
fi
if [ ! -f "$file" ]; then
    touch "$file"
fi
echo -n > "$file"

if [ "${SWITCH}" = 1 ]; then
    mpirun --allow-run-as-root -np ${WORLD_SIZE} --hostfile ${HOSTFILE} \
        --bind-to none --map-by slot \
        --mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
        --mca oob_tcp_if_include bond1 \
        -x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_switch.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/hetero/before.json,ds_parallel_config/hetero/after.json \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --json_file $JSON_FILE \
        --json_key $JSON_KEY \
        --vocab_file $VOCAB_FILE \
        --merge_file $MERGE_FILE \
        --vocab_size 30592 \
        --hidden_size $HIDDEN_SIZE \
        --ffn_hidden_size $FFN_HIDDEN_SIZE \
        --num_hidden_layers $NUM_LAYERS \
        --num_attention_heads $NUM_HEADS \
        --seq_length $SEQ_LEN \
        --epochs 4 \
        --steps 40 \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --hidden_act relu \
        --dropout_prob 0.1 \
        --bf16 \
        --use_flash_attn \
        --use_two_node \
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_pipeline \
        --hetero_data \
        --before_hetero_stages $BEFORE_STAGES_NUM_LIST \
        --before_micro_batch_num_list $BEFORE_MICRO_BATCH_NUM_LIST \
        --before_rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
        --before_unused_rank $BEFORE_UNUSED_RANK \
        --after_hetero_stages $AFTER_STAGES_NUM_LIST \
        --after_micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --after_rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --after_unused_rank $AFTER_UNUSED_RANK \
        --tencent
else
    mpirun --allow-run-as-root -np ${WORLD_SIZE} --hostfile ${HOSTFILE} \
        --bind-to none --map-by slot \
        --mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
        --mca oob_tcp_if_include bond1 \
        -x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/homo/dp${DP}_tp${TP}_pp${PP}.json,ds_parallel_config/hetero/after.json \
        --global_batch_size $GLOBAL_BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --json_file $JSON_FILE \
        --json_key $JSON_KEY \
        --vocab_file $VOCAB_FILE \
        --merge_file $MERGE_FILE \
        --vocab_size 30592 \
        --hidden_size $HIDDEN_SIZE \
        --ffn_hidden_size $FFN_HIDDEN_SIZE \
        --num_hidden_layers $NUM_LAYERS \
        --num_attention_heads $NUM_HEADS \
        --seq_length $SEQ_LEN \
        --epochs 4 \
        --steps 40 \
        --lr 1e-4 \
        --adam_weight_decay 0.01 \
        --hidden_act relu \
        --dropout_prob 0.1 \
        --bf16 \
        --use_flash_attn \
        --use_two_node \
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_stages $AFTER_STAGES_NUM_LIST \
        --hetero_pipeline \
        --hetero_data \
        --micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --unused_rank $AFTER_UNUSED_RANK \
        --tencent
fi