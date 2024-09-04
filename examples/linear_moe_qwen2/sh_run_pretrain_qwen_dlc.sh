export HF_ENDPOINT=https://hf-mirror.com

sh run_pretrain_qwen.sh  \
dlc  \
A1B   \
16    \
256 \
1e-5   \
1e-6   \
2048  \
2048  \
bf16  \
1   \
1  \
1 \
sel  \
true   \
false  \
false   \
false   \
100000  \
/cpfs01/user/sunweigao/my/data-SlimPajama/slimpajama_chunk1_chunk2_megatron_bin_data/mmap_qwen2_datasets_text_document  \
Qwen/Qwen2-0.5B  \
100000000000   \
10000   \
./output

# 运行run_pretrain_qwen.sh脚本，需要传入的参数列表如下
# ```
# ENV=$1                          # 运行环境: dlc, dsw
# MODEL_SIZE=$2                   # 模型结构参数量级：7B, 72B
# BATCH_SIZE=$3                   # 每卡训练一次迭代样本数: 4, 8
# GLOBAL_BATCH_SIZE=$4            # 全局batch size
# LR=$5                           # 学习率: 1e-5, 5e-5
# MIN_LR=$6                       # 最小学习率: 1e-6, 5e-6
# SEQ_LEN=$7                      # 序列长度
# PAD_LEN=$8                      # Padding长度：100
# PR=$9                           # 训练精度: fp16, bf16
# TP=${10}                        # 模型并行度
# PP=${11}                        # 流水并行度
# EP=${12}                        # 专家并行度
# AC=${13}                        # 激活检查点模式: sel, full
# DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
# FL=${15}                        # 是否使用Flash Attention: true, false
# SP=${16}                        # 是否使用序列并行: true, false
# TE=${17}                        # 是否使用Transformer Engine: true, false
# SAVE_INTERVAL=${18}             # 保存ckpt的间隔
# DATASET_PATH=${19}              # 训练数据集路径
# PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径 or 预训练模型的huggingface名字（会自动根据名字从huggingface下载tokenizer），这种情况从头训练而不是续训
# TRAIN_TOKENS=${21}              # 训练token数
# WARMUP_TOKENS=${22}             # 预热token数
# OUTPUT_BASEPATH=${23}           # 训练输出文件路径
# ```