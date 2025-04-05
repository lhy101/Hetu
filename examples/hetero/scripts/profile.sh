TP=${1:-2}
PP=${2:-2}
EXP_DIR=${3:-"./experiments"}
DP=${4:-1}

NUM_LAYERS=60
HIDDEN_SIZE=6656
NUM_HEADS=64
FFN_HIDDEN_SIZE=17920

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
SEQ_LEN=4096
SERVER_ADDR="30.203.138.189"
SERVER_PORT="23462"
HOST_FILE_PATH="/jizhicfs/pinxuezhao/lhy/hostfiles/host0123.yaml"
ENV_FILE_PATH="./scripts/env_H20.sh"

NUM_GPUS=$(expr $TP \* $PP \* $DP)
CP=1
DCP=${DP}
CP_LIST="["
for ((i=1; i<=DP; i++)); do
	if [ $i -ne 1 ]; then
		CP_LIST="$CP_LIST,"
	fi
	CP_LIST="$CP_LIST$CP"
done
CP_LIST="$CP_LIST]"
RECOMPUTE_LAYERS="[]"

echo run exp: tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 
# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/exp_tp${TP}_pp${PP}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/jizhicfs/pinxuezhao/lhy/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

python -m hetu.models.llama.generate_llama_4d_config \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
	--zero 

if [ ! -d "$EXP_DIR" ]; then
  mkdir -p "$EXP_DIR"
fi

CMD="python3 -u train_hetu_padding.py \
--compute_only 1 \
--torch_profile 0 \
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
--gpus_per_stage ${TP} \
--exp_file ${EXP_DIR}/layers60 "

source ${ENV_FILE_PATH}
python3 -m hetu.rpc.pssh_start \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
