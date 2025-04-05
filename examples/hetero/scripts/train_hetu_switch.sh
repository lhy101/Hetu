NUM_LAYERS=${1:-4}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-4}
MICRO_BATCH_SIZE=${6:-1}
FFN_HIDDEN_SIZE=${7:-11008}
# SERVER_ADDR="${IP_1}"
SERVER_ADDR="127.0.0.1" # 216
SERVER_PORT=${9:-"23457"}
HOST_FILE_PATH=${10:-"${ENV_PATH}/host_single.yaml"}
ENV_FILE_PATH=${11:-"${ENV_PATH}/env_A100.sh"}

NUM_GPUS=8
python generate_strategy_ds.py -p "./strategy" -m "llama"

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/hot-switch
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/combined_data.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

CMD="python3 -u train_hetu_switch.py \
--num_strategy=2 \
--ds_parallel_config ds_parallel_config/multi_strategy/strategy_1.json,ds_parallel_config/multi_strategy/strategy_2.json \
--strategy_config strategy/strategy_1.json,strategy/strategy_2.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
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
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
python3 -m hetu.rpc.pssh_start \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}