#!/bin/bash

export PATH=/cpfs01/user/dujusen/Linear-MoE-public/.venv/linear-moe/bin:$PATH
export LD_LIBRARY_PATH=/cpfs01/user/dujusen/Linear-MoE-public/.venv/linear-moe/lib:$LD_LIBRARY_PATH

set -e
ENV=dsw
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
LINEAR_MOE_PATH=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATH=${LINEAR_MOE_PATH}/third_party/Megatron-LM-0.4.0
export PYTHONPATH=${MEGATRON_PATH}:${LINEAR_MOE_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ENDPOINT=https://hf-mirror.com
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=1B
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=2048
PAD_LEN=2048
EXTRA_VOCAB_SIZE=0
PR=bf16
TP=1
PP=1
AC=sel
DO=true
FL=false
SP=false
TE=false
SAVE_INTERVAL=100000
DATASET_PATH=/cpfs01/shared/public/sunweigao/data-SlimPajama/slimpajama_chunk1_chunk2_megatron_bin_data/mmap_qwen2_datasets_text_document
PRETRAIN_CHECKPOINT_PATH=/cpfs01/user/dujusen/models/Llama-3.2-1B
TRAIN_TOKENS=10000000000
WARMUP_TOKENS=10000
OUTPUT_BASEPATH=./output

if [ $MODEL_SIZE = 8B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=8192

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

fi

if [ $MODEL_SIZE = 1B ]; then

NUM_LAYERS=16
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=8192
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="${ENV}-pretrain-megatron-gpt3-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 99,1,0 \
        --train-data-path ${DATASET_PATH} \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --dataset LLama-Pretrain-Idxmap \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --seed 1234 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type LLamaTokenizer \
        --swiglu \
        --normalization RMSNorm \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --norm-epsilon 1e-05 \
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_megatron_llama.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
