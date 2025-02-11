<div align="center">

# Linear-MoE

</div>

This repo offers Linear-MoE, a **production-ready framework** for modeling and training Linear-MoE models, non-invasively built on the latest [Megatron-Core](https://github.com/NVIDIA/Megatron-LM). **Contributions through pull requests are highly encouraged!**

<!-- It supports state-of-the-art open-source Mixture of Experts (MoE) models, seamlessly integrated with advanced linear sequence modeling techniques such as Linear Attention, State Space Modeling, and Linear RNN. LInear-MoE is still under development, **Contributions through pull requests are highly encouraged!** -->

# Model Matrix

|   Linear Sequence Modeling  |  Instance  |  Qwen2 MoE (@Alibaba)  |    Deepseek v2 MoE (@Deepseek)       |    Mixtral MoE (@Mistral AI)   | Llama3 (@Meta)   |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Linear Attention (LA) |       [Basic Linear Attention](https://arxiv.org/abs/2006.16236) <br> (@Idiap@EPFL)  | ✅ |    ✅    |   ✅   | ✅   |
|  |       [Lightning Attention](https://arxiv.org/abs/2405.17381) <br> (@Shanghai AI Lab)       | ✅ |          ✅          |     ✅      | ✅   |
|  |       [Retention](https://arxiv.org/abs/2307.08621) <br> (@MSRA@THU)       | ✅ |          ✅          |     ✅      | ✅   |
|  |         [GLA](https://arxiv.org/abs/2312.06635)  <br> (@MIT@IBM)         | ✅ |     ✅      |    ✅       | ✅   |
|  |           [Delta Net](https://arxiv.org/abs/2102.11174) <br> (@MIT)            | ✅ |    ✅    |     ✅      | ✅   |
|  |           [GSA](https://arxiv.org/abs/2409.07146) <br> (@SUDA@MIT)      | ✅ |    ✅    |     ✅      | ✅   |
|  | [Based](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based) <br> (@Stanford) | ✅ |      ✅      |      ✅     |  ✅   |
|  |            [Rebased](https://arxiv.org/abs/2402.10644) <br> (@Tinkoff)            | ✅ |  ✅  |      ✅     | ✅   |
|  |            [LASP-2](https://arxiv.org/abs/) <br> (@Shanghai AI Lab)            | ✅ |  ✅  |      ✅     | ✅   |
| State Space Modeling (SSM) |             [Mamba2](https://arxiv.org/abs/2405.21060) <br> (@Princeton@CMU) | ✅ | ✅  |   ✅   | ✅   |
| Linear RNN |             [RWKV6](https://arxiv.org/abs/2404.05892) <br> (@RWKV)              |  ✅  |   ✅   |    ✅    | ✅   |
|  |             [HGRN2](https://arxiv.org/abs/2404.07904) <br> (@TapTap@Shanghai AI Lab)             | ✅ |   ✅   |   ✅   |  ✅   |
| Softmax Attention |             [Softmax Attention](https://arxiv.org/abs/1706.03762) <br> (@Google)             | ✅ |   ✅   |   ✅   |  ✅   |
|  |             [FlashAttention-2](https://arxiv.org/abs/2307.08691) <br> (@Princeton@Stanford)             | ✅ |   ✅   |   ✅   |  ✅   |


# Framework Overview

<p align="center">
  <img src="./images/linear-moe-fig1.png" width="80%" />
  <figcaption style="text-align: center;">Figure 1: Linear-MoE Framework</figcaption>
</p>

<p align="center">
  <img src="./images/linear-moe-fig2.png" width="80%" />
  <figcaption style="text-align: center;">Figure 2: Linear-MoE Model Architecture</figcaption>
</p>

# Installation

Your environment should satify the following requirements:

- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2

## Virtualenv

```bash
# create a conda env, install PyTorch
conda create -n linear-moe python=3.11
conda activate linear-moe
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# (if needed) Apex
git clone https://github.com/NVIDIA/apex.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# (if needed) FlashAttention
MAX_JOBS=8 pip install flash-attn --no-build-isolation

# (if needed) dropout_layer_norm in FlashAttention
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/csrc/layer_norm & pip install .

# Transformer Engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Linear-MoE 
git clone --recurse-submodules https://github.com/OpenSparseLLMs/Linear-MoE.git

# requirements
pip install -r requirements.txt
```


## Container
We recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with DGX nodes, which already have relatively new versions of CUDA, cuDNN, NCCL, PyTorch, Triton, Apex, TransformerEngine, etc., installed.

On the top of NGC's PyTorch container, you can setup Linear-MoE with:
```bash
# Linear-MoE 
git clone --recurse-submodules https://github.com/OpenSparseLLMs/Linear-MoE.git

# requirements
pip install -r requirements.txt
```


# Usage

## Pretraining or Finetuning

<!-- **Key Features related to pretraining in Linear-MoE**
- Multiple linear sequence modeling options (Linear Attention, SSM, Linear RNN)
- Flexible MoE configurations
- Multi-node distributed training
- Mixed precision training
- Gradient checkpointing
- Token dropping for efficient MoE training -->

To pretrain or finetune a Linear-MoE model, you can:

1. Open `examples`, choose the model you are going to pretrain or finetune, e.g. `linear_moe_qwen2`.

2. Edit `run_pretrain_qwen.sh` or `run_finetune_qwen.sh` to set your configurations, like:
- Model size (e.g., 0.5B, 1.5B, 7B)
- Batch size
- Learning rate
- Model architecture (e.g., LSM modules, number of experts)
- Distributed training settings (TP, PP, CP, EP sizes)


3. **Start pretraining or finetuning** by: `sh run_pretrain_qwen.sh` or `sh run_finetune_qwen.sh`.


## Evaluation

We use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for benchmark evaluation. See [eval/README.md](eval/README.md) for detailed instruction.


# Acknowledgement
We built this repo upon [alibaba/PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch), and take [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) as the training engine. We use the triton-implemented linear attention kernels from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and CUDA implemented Mamba2 kernel from [state-spaces/mamba](https://github.com/state-spaces/mamba) to accelerate the execution.

# Citation
If you find this repo useful, please consider citing our work:
```bib

```