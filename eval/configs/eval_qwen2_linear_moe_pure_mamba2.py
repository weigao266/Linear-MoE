from mmengine.config import read_base


with read_base():
    from opencompass.configs.datasets.wikitext.wikitext_103_raw_ppl import wikitext_103_raw_datasets
    from opencompass.configs.datasets.lambada.lambada_gen import lambada_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets # ARC-easy
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets # ARC-challenge

datasets = wikitext_103_raw_datasets + lambada_datasets + piqa_datasets + \
      hellaswag_datasets + winogrande_datasets + ARC_e_datasets + ARC_c_datasets

from eval.models.qwen2_linear_moe_pure_mamba2 import Qwen2LinearMoePureMamba2

LINEAR_MOE_PATH = "/cpfs01/user/sunweigao/landisen/Linear-MoE-public"
CHECKPOINT_PATH = f"{LINEAR_MOE_PATH}/checkpoint/pretrain-mcore-pure_mamba2-qwen2-A1B-lr-1e-5-minlr-1e-6-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-100000000000-wt-10000"

models = [
    dict(
        type=Qwen2LinearMoePureMamba2,
        path=CHECKPOINT_PATH, #'huggyllama/llama-7b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_path=CHECKPOINT_PATH, # 'huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        max_out_len=512,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=1,
    )
]

