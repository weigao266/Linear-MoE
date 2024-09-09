
<div align="center">

# Linear-MoE

[![hf_model](https://img.shields.io/badge/🤗-Models-blue.svg)](https://huggingface.co/xxx) | [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/xxx)
</div>

This repository offers a **production-ready framework** for modeling and training Linear-MoE models, non-invasively built on the latest [Megatron-Core](https://github.com/NVIDIA/Megatron-LM). It supports state-of-the-art open-source Mixture of Experts (MoE) models, seamlessly integrated with advanced linear sequence modeling techniques such as Linear Attention, State Space Models, and Linear RNNs. **Contributions through pull requests are highly encouraged!**

<!-- <div align="center">
  <img width="400" alt="image" src="https://github.com/xxx">
</div> -->

# Models

|   Linear Sequence Modeling  |  Instance  |  Qwen2 MoE (@Alibaba)  |    Deepseek v2 MoE (@Deepseek)       |    Mixtral MoE (@Mistral AI)   |
| :-----: | :----------------------------: | :----------------------------: | :---------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| Linear Attention |       [Basic Linear Attention](https://arxiv.org/abs/2006.16236) <br> (@Idiap@EPFL)       | ✅ |          ✅          |     ✅      |
|  |       [Retention](https://arxiv.org/abs/2307.08621) <br> (@MSRA@THU)       | ✅ |          ✅          |     ✅      |
|  |         [GLA](https://arxiv.org/abs/2312.06635)  <br> (@MIT@IBM)         | ✅ |     ✅      |    ✅       |
|  |           [Delta Net](https://arxiv.org/abs/2102.11174) <br> (@MIT)            | ✅ |    ✅    |     ✅      |
|  | [Based](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based) <br> (@Stanford@HazyResearch) | ✅ |      ✅      |      ✅     | 
|  |            [Rebased](https://arxiv.org/abs/2402.10644) <br> (@Tinkoff)            | ✅ |  ✅  |      ✅     |
| State Space Modeling (SSM) |             [Mamba2](https://arxiv.org/abs/2405.21060) <br> (@Princeton@CMU)              | ✅ | ✅  |      |
| Linear RNN |             [RWKV6](https://arxiv.org/abs/2404.05892) <br> (@RWKV)              |  ✅  |   ✅   |    ✅    |
|  |             [HGRN2](https://arxiv.org/abs/2404.07904) <br> (@TapTap@Shanghai AILab)             | ✅ |   ✅   |   ✅   |  



# Installation

- [PyTorch](https://pytorch.org/) >= 2.0
- [Triton](https://github.com/openai/triton) >=2.2
- [einops](https://einops.rocks/)


# Usage

## Pretraining

## Finetuning

## Generation

## Evaluation


# Citation
<!-- If you find this repo useful, please consider citing our works:
```bib

``` -->