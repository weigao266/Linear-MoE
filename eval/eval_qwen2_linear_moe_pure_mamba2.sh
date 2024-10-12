#!/bin/bash
# bash run_text_generation_megatron_llama.sh dsw /workspace/PAI-Megatron-Patch /mnt/llama-ckpts/Ziya-LLaMA-13B-to-megatron-tp1-pp1 13B 1 1 1024 80 0 fp16 0 512 512 /mnt/llama-datasets/gen.jsonl /mnt/llama-datasets/cn_output.txt 0.85 1 1
set -e
ENV=dsw #$1
export CUDA_VISIBLE_DEVICES=0,1
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# GPUS_PER_NODE=2
# NNODES=1
# NODE_RANK=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
LINEAR_MOE_PATH=/cpfs01/user/sunweigao/landisen/Linear-MoE-public #$2
MEGATRON_PATH=${LINEAR_MOE_PATH}/third_party/Megatron-LM-0.9.0 #${LINEAR_MOE_HPAT}/Megatron-LM-231007
echo $MEGATRON_PATH
export PYTHONPATH=${MEGATRON_PATH}:${LINEAR_MOE_PATH}:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=${LINEAR_MOE_PATH}/checkpoint/pretrain-mcore-pure_mamba2-qwen2-A1B-lr-1e-5-minlr-1e-6-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-100000000000-wt-10000 #$3
MODEL_SIZE=A1B #$4  #7B, 13B, 70B
TP=1 #$5
BS=1 #$6
SEQ_LEN=2048 #$7
PAD_LEN=2048 #$8
EXTRA_VOCAB_SIZE=293 #$9
PR=bf16 #${10}
TOP_K=10 #${11}
INPUT_SEQ_LEN=2048 #${12}
OUTPUT_SEQ_LEN=2048 #${13}
INPUT_FILE=${LINEAR_MOE_PATH}/examples/linear_moe_qwen2/input_file.json #${14}
OUTPUT_FILE=${LINEAR_MOE_PATH}/examples/linear_moe_qwen2/output_file.txt #${15}
TOP_P=0 #${16}
TEMPERATURE=1.0 #${17}
# set this penalty between 1.1 and 1.5 to reduce repetition, default is 1.2
REPETITION_PENALTY=1.2 #${18}

# ENV=dlc
MODEL_SIZE=A1B
BATCH_SIZE=256
GLOBAL_BATCH_SIZE=256
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=2048
PAD_LEN=2048
PR=bf16
TP=1
PP=1
EP=1
AC=sel
DO=true
FL=false
SP=false
TE=false
SAVE_INTERVAL=100000
# DATASET_PATH=input_file.txt #/cpfs01/user/sunweigao/my/data-SlimPajama/slimpajama_chunk1_chunk2_megatron_bin_data/mmap_qwen2_datasets_text_document
# PRETRAIN_CHECKPOINT_PATH=${LINEAR_MOE_PATH}/checkpoint/pretrain-mcore-pure_mamba2-qwen2-A1B-lr-1e-5-minlr-1e-6-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-100000000000-wt-10000 #/cpfs01/user/sunweigao/my/qwen-ckpts/Qwen2-0.5B
TRAIN_TOKENS=100000000000
WARMUP_TOKENS=10000
OUTPUT_BASEPATH=./output

LA_MODULE="pure_mamba2"
BASE_MODEL="qwen2"
HYBRID_ATTENTION_RATIO=0.0
HYBRID_MLP_RATIO=0.0

# SSM
linear_moe_options=" \
        --use-la-module \
        --la-module ${LA_MODULE} \
        --base-model ${BASE_MODEL} \
        "

# Linear Attention
# linear_moe_options=" \
#         --use-la-module \
#         --la-module ${LA_MODULE} \
#         --la-mode chunk \
#         --base-model ${BASE_MODEL} \
#         --la-feature-map swish \
#         --la-output-norm rmsnorm \
#         --la-gate-fn swish \
#         "

# # Linear RNN
# linear_moe_options=" \
#         --use-la-module \
#         --la-module ${LA_MODULE} \
#         --la-mode chunk \
#         --base-model ${BASE_MODEL} \
#         --la-output-norm rmsnorm \
#         --la-gate-fn swish \
#         "

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

if [ $MODEL_SIZE = 0.5B ]; then

HIDDEN_SIZE=512 # change it form 896 to 512, to run mamba2
INTERMEDIATE_SIZE=4864
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=24
NUM_ATTENTION_HEADS=16 # change it form 14 to 16, to run mamba2
NUM_HIDDEN_LAYERS=24
# NUM_KEY_VALUE_HEADS=2
# linear_moe: set NUM_KEY_VALUE_HEADS = NUM_ATTENTION_HEADS in linear attention
NUM_KEY_VALUE_HEADS=16 # change it form 14 to 16, to run mamba2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

elif [ $MODEL_SIZE = 0.5Boriginal ]; then

HIDDEN_SIZE=896
INTERMEDIATE_SIZE=4864
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=24
NUM_ATTENTION_HEADS=14
NUM_HIDDEN_LAYERS=24
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "


elif [ $MODEL_SIZE = 1.5B ]; then

HIDDEN_SIZE=1536
INTERMEDIATE_SIZE=8960
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=12
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

elif [ $MODEL_SIZE = 7B ]; then

HIDDEN_SIZE=3584
INTERMEDIATE_SIZE=18944
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=28
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=4
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=421

moe_options=" \
            "

elif [ $MODEL_SIZE = 72B ]; then

HIDDEN_SIZE=8192
INTERMEDIATE_SIZE=29568
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=80
NUM_ATTENTION_HEADS=64
NUM_HIDDEN_LAYERS=80
NUM_KEY_VALUE_HEADS=8
RMS_NORM_EPS=1e-5
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=421

moe_options=" \
            "

elif [ $MODEL_SIZE = A1B ]; then

HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=1024
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=16
MOE_INTERMEDIATE_SIZE=1024
NUM_ATTENTION_HEADS=16
NUM_EXPERTS=64
NUM_EXPERTS_PER_TOPK=8
NUM_HIDDEN_LAYERS=16
NUM_KEY_VALUE_HEADS=16
RMS_NORM_EPS=1e-6
ROPE_THETA=10000
SHARED_EXPERT_INTERMEDIATE_SIZE=0
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            --moe-router-topk ${NUM_EXPERTS_PER_TOPK} \
            --num-experts ${NUM_EXPERTS} \
            --expert-model-parallel-size ${EP}\
            --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
            --shared-moe-ffn-hidden-size ${SHARED_EXPERT_INTERMEDIATE_SIZE}"

elif [ $MODEL_SIZE = A14B ]; then

HIDDEN_SIZE=3584
INTERMEDIATE_SIZE=18944
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
MOE_INTERMEDIATE_SIZE=2560
NUM_ATTENTION_HEADS=28
NUM_EXPERTS=64
NUM_EXPERTS_PER_TOPK=8
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=4
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SHARED_EXPERT_INTERMEDIATE_SIZE=20480
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            --moe-router-topk ${NUM_EXPERTS_PER_TOPK} \
            --num-experts ${NUM_EXPERTS} \
            --expert-model-parallel-size ${EP}\
            --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
            --shared-moe-ffn-hidden-size ${SHARED_EXPERT_INTERMEDIATE_SIZE} \
            --enable-shared-expert"

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
		    --fp16 \
        --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
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

# if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
#     load_options=" \
#             --load $PRETRAIN_CHECKPOINT_PATH"
# fi

if [ $CHECKPOINT_PATH != none ]; then
    load_options=" \
		    --load $CHECKPOINT_PATH"
fi

if [ $INPUT_FILE = none ]; then
    input_options=" \
		               "
else
    input_options=" \
        --text-generate-output-file ${OUTPUT_FILE}\
        --text-generate-input-file ${INPUT_FILE} \
        "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="pretrain-mcore-${LA_MODULE}-qwen2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
# mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
# mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
# mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
# mkdir -p ${TENSORBOARD_DIR}

LOG_FILE="${OUTPUT_BASEPATH}/log/${current_time}_${NAME}.log"

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --data-path ${DATASET_PATH} \
        --split 99,1,0 \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_HIDDEN_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 10 \
        --eval-interval 100000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-log-interval 10 \
        --tensorboard-queue-size 1000 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --add-qkv-bias \
        --group-query-attention \
        --num-query-groups ${NUM_KEY_VALUE_HEADS} \
        --rotary-percent 1.0 \
        --rotary-base ${ROPE_THETA} \
        --rotary-seq-len-interpolation-factor 1 \
        --no-create-attention-mask-in-dataloader \
        --hybrid-attention-ratio ${HYBRID_ATTENTION_RATIO} \
        --hybrid-mlp-ratio ${HYBRID_MLP_RATIO} \
        "

# rapidformer_options="  \
#         --micro-batch-size ${BS} \
#         --num-layers ${NUM_LAYERS}  \
#         --hidden-size ${HIDDEN_SIZE}  \
#         --num-attention-heads ${NUM_ATTN_HEADS}  \
#         --ffn-hidden-size ${INTERMEDIATE_SIZE} \
#         --seq-length ${SEQ_LEN} \
#         --max-position-embeddings ${SEQ_LEN} \
#         --tensor-model-parallel-size ${TP} \
#         --pipeline-model-parallel-size 1 \
#         --no-load-optim \
#         --no-load-rng \
#         --top-p ${TOP_P} \
#         --temperature ${TEMPERATURE}  \
#         --top-k ${TOP_K} \
#         --input-len ${INPUT_SEQ_LEN} \
#         --out-seq-length ${OUTPUT_SEQ_LEN}  \
#         --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
#         --max-padding-length ${PAD_LEN} \
#         --use-distributed-optimizer \
#         --swiglu \
#         --use-llama2-rotary-position-embeddings \
#         --position-embedding-type rope \
#         --untie-embeddings-and-output-weights \
#         --patch-tokenizer-type LLamaTokenizer \
#         --normalization RMSNorm \
#         --repetition-penalty ${REPETITION_PENALTY} \
#         --disable-bias-linear
#     "

run_cmd="torchrun $DISTRIBUTED_ARGS run.py configs/eval_qwen2_linear_moe_pure_mamba2.py -w outputs/qwen2_linear_moe --debug
 ${megatron_options} ${pr_options} ${load_options} ${input_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${moe_options} ${linear_moe_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
