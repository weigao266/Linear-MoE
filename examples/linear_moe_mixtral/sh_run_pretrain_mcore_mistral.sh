export HF_ENDPOINT=https://hf-mirror.com

sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
Small   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
true \
100000  \
/cpfs01/user/sunweigao/my/mistral-datasets/wudao_mistralbpe_content_document \
mistralai/Mistral-7B-v0.1 \
100000000   \
10000   \
./output_mcore_mistral

# 运行run_pretrain_mcore_mistral.sh脚本，需要传入的参数列表如下
# ```
# ENV=$1                          # 运行环境: dlc, dsw
# LINEAR_MOE_PATH=$2          # 设置LINEAR_MOE的代码路径
# MODEL_SIZE=$3                   # 模型结构参数量级：7B, 13B
# BATCH_SIZE=$4                   # 每卡训练一次迭代样本数: 4, 8
# GLOBAL_BATCH_SIZE=$5            # 全局batch size
# LR=$6                           # 学习率: 1e-5, 5e-5
# MIN_LR=$7                       # 最小学习率: 1e-6, 5e-6
# SEQ_LEN=$8                      # 序列长度
# PAD_LEN=$9                      # Padding长度：100
# EXTRA_VOCAB_SIZE=${10}          # 词表扩充大小
# PR=${11}                        # 训练精度: fp16, bf16
# TP=${12}                        # 模型并行度
# PP=${13}                        # 流水并行度
# AC=${14}                        # 激活检查点模式: sel, full
# DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
# FL=${16}                        # 是否使用Flash Attention: true, false
# SP=${17}                        # 是否使用序列并行: true, false
# TE=${18}                        # 是否使用Transformer Engine: true, false
# MOE=${19}                       # 是否打开MOE: true, false
# SAVE_INTERVAL=${20}             # 保存ckpt的间隔
# DATASET_PATH=${21}              # 训练数据集路径
# PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
# TRAIN_TOKENS=${23}              # 训练token数
# WARMUP_TOKENS=${24}             # 预热token数
# OUTPUT_BASEPATH=${25}           # 训练输出文件路径