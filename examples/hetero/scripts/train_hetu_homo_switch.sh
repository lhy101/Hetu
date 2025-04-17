NUM_LAYERS=${1:-60}
HIDDEN_SIZE=${2:-6656}
NUM_HEADS=${3:-64}
SEQ_LEN=${4:-4096}
GLOBAL_BATCH_SIZE=${5:-4}
MICRO_BATCH_SIZE=${6:-1}
FFN_HIDDEN_SIZE=${7:-17920}
# SERVER_ADDR="${IP_1}"
SERVER_ADDR="30.203.138.189"
SERVER_PORT=${9:-"23457"}
HOST_FILE_PATH=${10:-"/jizhicfs/pinxuezhao/lhy/hostfiles/host0123.yaml"}
ENV_FILE_PATH=${11:-"./scripts/env_H20.sh"}

NUM_GPUS=32
python generate_strategy_ds.py -p "./homo_strategy" -m "llama"

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/homo-hot-switch
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/jizhicfs/pinxuezhao/lhy/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

CMD="python3 -u train_hetu_switch.py \
--num_strategy=2 \
--ds_parallel_config ds_parallel_config/homo_multi_strategy/strategy_2.json,ds_parallel_config/homo_multi_strategy/strategy_3.json \
--strategy_config homo_strategy/strategy_2.json,homo_strategy/strategy_3.json \
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
