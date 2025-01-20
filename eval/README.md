# Linear-MoE Evaluation

First you should install our Linear-MoE-adapted `lm-evaluation-harness`:

```bash
cd ./third_party/lm-evaluation-harness
pip install -e .
```

Set your Linear-MoE checkpoint path in `lm_eval_linear_moe.sh`:

```bash
CHECKPOINT_DIR=/your/checkpoint/dir
CHECKPOINT_PATH=${CHECKPOINT_DIR}/pretrain-mcore-linear_attention-qwen2-A0.3B-lr-1e-4-minlr-1e-5-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-15000000000-wt-10000
```

and modify the model configuration correspondingly:

```bash
MODEL_SIZE=A0.3B
SEQ_LEN=2048
PAD_LEN=2048
PR=bf16
TP=1
PP=1
EP=1
...
LA_MODULE="linear_attention"
BASE_MODEL="qwen2"
# for models except mamba2
LAYER_TYPE_LIST="LLLLLLLLLLLL" # LAYER_TYPE_LIST="LLLNLLLNLLLN" if hybrid model
# Linear Attention
linear_moe_options=" \
        --use-la-module \
        --la-module ${LA_MODULE} \
        --la-mode fused_chunk \
        --base-model ${BASE_MODEL} \
        --la-feature-map elu \
        --la-output-norm rmsnorm \
        --la-gate-fn swish \
        --layer-type-list ${LAYER_TYPE_LIST} \
        "
```

Here you can set the evaluation task and `lm-evaluation-harness` configuration:

```bash
run_cmd="torchrun $DISTRIBUTED_ARGS --no-python lm_eval \
 --model linear_moe \
 --model_args path=${CHECKPOINT_PATH} max_length=2048 \
 --tasks piqa \
 --device cuda \
 --batch_size 16 \
 --output_path lm_eval_result \
 ${megatron_options} ${pr_options} ${load_options} ${input_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${moe_options} ${linear_moe_options}"
```

Evaluate Linear-MoE by:

```bash
sh lm_eval_linear_moe.sh
```

The evaluation results would be presented as follows:

```bash
linear_moe (path=/cpfs04/shared/MOE/landisen/models/linear_moe_checkpoints/pretrain-mcore-linear_attention-qwen2-A0.3B-lr-1e-4-minlr-1e-5-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-15000000000-wt-10000), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
|Tasks|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----|------:|------|-----:|--------|---|-----:|---|-----:|
|piqa |      1|none  |     0|acc     |↑  |0.6436|±  |0.0112|
|     |       |none  |     0|acc_norm|↑  |0.6436|±  |0.0112|
```
