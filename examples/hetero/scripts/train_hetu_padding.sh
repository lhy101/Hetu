NUM_LAYERS=${1:-60}
# NUM_LAYERS=${1:-80}
# NUM_LAYERS=${1:-4}
HIDDEN_SIZE=${2:-6144}
HIDDEN_SIZE=${2:-6656}
# HIDDEN_SIZE=${2:-8192}
# HIDDEN_SIZE=${2:-256}
# HIDDEN_SIZE=${2:-16}
NUM_HEADS=${3:-64}
# NUM_HEADS=${3:-2}
# SEQ_LEN=${4:-1024}
SEQ_LEN=${4:-4096}
# GLOBAL_BATCH_SIZE=${5:-16}
GLOBAL_BATCH_SIZE=${5:-4}
MICRO_BATCH_SIZE=${6:-1}
FFN_HIDDEN_SIZE=${7:-17904}
FFN_HIDDEN_SIZE=${7:-17920}
# FFN_HIDDEN_SIZE=${7:-49152}
# FFN_HIDDEN_SIZE=${7:-28672}
# FFN_HIDDEN_SIZE=${7:-2752}
# FFN_HIDDEN_SIZE=${7:-16}
SERVER_ADDR="30.203.138.189"
# SERVER_ADDR="30.203.136.4"
SERVER_PORT=${8:-"23456"}
HOST_FILE_PATH=${9:-"/jizhicfs/pinxuezhao/lhy/hostfiles/host012345.yaml"}
ENV_FILE_PATH=${10:-"./scripts/env_H20.sh"}

COMPUTE_ONLY=0
TORCH_PROFILE=1
CASE=1
if [[ ${CASE} -eq 0 ]]; then
	HETERO=false
	NUM_GPUS=16
	TP=4
	PP=4
	DP=1
	CP=1
elif [[ ${CASE} -eq 1 ]]; then
	HETERO=false
	NUM_GPUS=16
	TP=8
	PP=1
	DP=1
	CP=2
elif [[ ${CASE} -eq 2 ]]; then
# 32*H20+16*H800
	HETERO=true
	NUM_GPUS=48
	DP=3
	CP_LIST="[1,1,1]"
	TP=4
	LAYERS_NUM_LIST="[[15,15,15,15],[15,15,15,15],[15,15,15,15]]"
	MICRO_BATCH_NUM_LIST="[12,12,40]"
	UNUSED_RANK="[]"
	RANK_TO_DEVICE_MAPPING="null"
	SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 3 ]]; then
# 32*H20+16*H800
        HETERO=true
        NUM_GPUS=48
        DP=4
        CP_LIST="[1,1,1,1]"
        TP=4
        LAYERS_NUM_LIST="[[11,11,38],[11,11,38],[11,11,38],[11,11,38]]"
        MICRO_BATCH_NUM_LIST="[16,16,16,16]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:32,9:33,10:34,11:35,12:8,13:9,14:10,15:11,16:12,17:13,18:14,19:15,20:36,21:37,22:38,23:39,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:40,33:41,34:42,35:43,36:24,37:25,38:26,39:27,40:28,41:29,42:30,43:31,44:44,45:45,46:46,47:47}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 4 ]]; then
# 16*H20+16*H800
        HETERO=true
        NUM_GPUS=32
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[7,7,23,23],[7,7,23,23]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:16,9:17,10:18,11:19,12:20,13:21,14:22,15:23,16:8,17:9,18:10,19:11,20:12,21:13,22:14,23:15,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 5 ]]; then
# 24*H20+8*H800
        HETERO=true
        NUM_GPUS=32
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[9,10,10,31],[9,10,10,31]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:24,13:25,14:26,15:27,16:12,17:13,18:14,19:15,20:16,21:17,22:18,23:19,24:20,25:21,26:22,27:23,28:28,29:29,30:30,31:31}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 6 ]]; then
# 24*H20+16*H800
        HETERO=true
        NUM_GPUS=40
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[6,6,6,21,21],[6,6,6,21,21]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:24,13:25,14:26,15:27,16:28,17:29,18:30,19:31,20:12,21:13,22:14,23:15,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:32,33:33,34:34,35:35,36:36,37:37,38:38,39:39}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 7 ]]; then
# 24*H20+15*H800
        HETERO=true
        NUM_GPUS=44
        DP=3
        CP_LIST="[1,1,1]"
        TP=4
        LAYERS_NUM_LIST="[[7,7,23,23],[11,11,38],[12,11,24,13]]"
        MICRO_BATCH_NUM_LIST="[29,18,17]"
        UNUSED_RANK="[38,39,41,42,43]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:24,9:25,10:26,11:27,12:28,13:29,14:30,15:31,16:8,17:9,18:10,19:11,20:12,21:13,22:14,23:15,24:32,25:33,26:34,27:35,28:16,29:17,30:18,31:19,32:20,33:21,34:22,35:23,36:36,37:37,38:39,39:40,40:38,41:41,42:42,43:43}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 8 ]]; then
# 24*H20+15*H800
        HETERO=true
        NUM_GPUS=44
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[6,6,6,21,21],[6,6,6,22,13,7]]"
        MICRO_BATCH_NUM_LIST="[33,31]"
        UNUSED_RANK="[38,39,41,42,43]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:24,13:25,14:26,15:27,16:28,17:29,18:30,19:31,20:12,21:13,22:14,23:15,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:32,33:33,34:34,35:35,36:36,37:37,38:39,39:40,40:38,41:41,42:42,43:43}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 9 ]]; then
# 32*H20+16*H800
        HETERO=true
        NUM_GPUS=48
        DP=2
        CP_LIST="[1,1]"
        TP=8
        LAYERS_NUM_LIST="[[17,17,46],[17,17,46]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:32,17:33,18:34,19:35,20:36,21:37,22:38,23:39,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:24,33:25,34:26,35:27,36:28,37:29,38:30,39:31,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 10 ]]; then
# 24*H20+16*H800
        HETERO=true
        NUM_GPUS=40
        DP=1
        CP_LIST="[1]"
        TP=8
        LAYERS_NUM_LIST="[[10,10,10,25,25]]"
        MICRO_BATCH_NUM_LIST="[64]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="null"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 11 ]]; then
# 16*H20+16*H800
        HETERO=true
        NUM_GPUS=32
        DP=1
        CP_LIST="[1]"
        TP=8
        LAYERS_NUM_LIST="[[11,11,29,29]]"
        MICRO_BATCH_NUM_LIST="[64]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="null"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 12 ]]; then
# 24*H20+8*H800
        HETERO=true
        NUM_GPUS=32
        DP=1
        CP_LIST="[1]"
        TP=8
        LAYERS_NUM_LIST="[[16,16,16,32]]"
        MICRO_BATCH_NUM_LIST="[64]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="null"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 13 ]]; then
# 32*H20+16*H800
# elastic-1
        HETERO=true
        NUM_GPUS=48
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[5,6,6,6,18,19],[5,6,6,6,18,19]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:32,9:33,10:34,11:35,12:8,13:9,14:10,15:11,16:12,17:13,18:14,19:15,20:36,21:37,22:38,23:39,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:40,33:41,34:42,35:43,36:24,37:25,38:26,39:27,40:28,41:29,42:30,43:31,44:44,45:45,46:46,47:47}"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:32,17:33,18:34,19:35,20:36,21:37,22:38,23:39,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:24,33:25,34:26,35:27,36:28,37:29,38:30,39:31,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 14 ]]; then
# 32*H20
        HETERO=true
        NUM_GPUS=32
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[15,15,15,15],[15,15,15,15]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 15 ]]; then
# 31*H20
        HETERO=true
        NUM_GPUS=31
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[15,15,15,15],[16,16,16,8,4]]"
        MICRO_BATCH_NUM_LIST="[33,31]"
        UNUSED_RANK="[30,31,33,34,35]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31}"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:31,31:32,32:30,33:33,34:34,35:35}"
        SEQ_LEN_LIST="null"
elif [[ ${CASE} -eq 16 ]]; then
# 24*H20
        HETERO=true
        NUM_GPUS=24
        DP=2
        CP_LIST="[1,1]"
        TP=4
        LAYERS_NUM_LIST="[[20,20,20],[20,20,20]]"
        MICRO_BATCH_NUM_LIST="[32,32]"
        UNUSED_RANK="[]"
        RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23}"
        SEQ_LEN_LIST="null"
else
    echo unknown CASE
	exit 1
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/jizhicfs/pinxuezhao/lhy/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# homo
if [ "${HETERO}" = false ]; then

CP_LIST="["
for ((i=1; i<=DP; i++)); do
	if [ $i -ne 1 ]; then
		CP_LIST="$CP_LIST,"
	fi
	CP_LIST="$CP_LIST$CP"
done
CP_LIST="$CP_LIST]"

python -m hetu.models.llama.generate_llama_4d_config \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
        --zero

CMD="python3 -u train_hetu_padding.py \
--compute_only $COMPUTE_ONLY \
--torch_profile $TORCH_PROFILE \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/llama_homo/dp${DP}_cp${CP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--cp_list \"${CP_LIST}\" \
--gpus_per_stage ${TP}"

# hetero
else

python -m hetu.models.llama.generate_llama_hetero_4d_config \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp_list $CP_LIST \
	--tp $TP \
	--hetero_layers $LAYERS_NUM_LIST \
	--rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
	--unused_rank $UNUSED_RANK \
	--file_name "/jizhicfs/pinxuezhao/lhy/HSPMD/examples/hetero/ds_parallel_config/llama_hetero/hetero_config.json"

CMD="python3 -u train_hetu_padding.py \
--compute_only $COMPUTE_ONLY \
--torch_profile $TORCH_PROFILE \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/llama_hetero/hetero_config.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--cp_list \"${CP_LIST}\" \
--hetero \
--seq_len_list \"${SEQ_LEN_LIST}\" \
--gpus_per_stage ${TP} \
--hetero_layers \"${LAYERS_NUM_LIST}\" \
--micro_batch_num_list \"${MICRO_BATCH_NUM_LIST}\" \
--rank_to_device_mapping \"${RANK_TO_DEVICE_MAPPING}\" \
--unused_rank \"${UNUSED_RANK}\""

fi

source ${ENV_FILE_PATH}
python3 -m hetu.rpc.pssh_start \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
