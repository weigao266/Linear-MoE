from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.wikitext.wikitext_103_raw_ppl import wikitext_103_raw_datasets
    from opencompass.configs.datasets.lambada.lambada_gen import lambada_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets # ARC-easy
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets # ARC-challenge

datasets = hellaswag_datasets

from eval.models.qwen2_linear_moe import Qwen2LinearMoe

models = [
    dict(
        type=Qwen2LinearMoe,
        path="", #'huggyllama/llama-7b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_path="", # 'huggyllama/llama-7b',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        max_out_len=512,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=32,
    )
]

