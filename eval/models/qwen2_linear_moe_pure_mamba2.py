import os
import sys
import math
from typing import Optional, Dict, List, Union
path_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.join(path_dir, "opencompass"))
# sys.path.append(os.path.join(path_dir, "third_party/Megatron-LM-0.9.0"))
from opencompass.models.base import BaseModel, LMTemplateParser

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import recv_forward
from megatron.core.pipeline_parallel.p2p_communication import send_forward
import megatron.legacy.model
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group,
    unwrap_model
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import parallel_state, tensor_parallel
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model

from linear_moe.model.qwen2.model import GPTModel
from linear_moe.generation.api import generate
from linear_moe.data.utils import get_batch_on_this_tp_rank_original
from linear_moe.model.qwen2.layer_specs import get_gpt_layer_with_transformer_engine_spec,get_gpt_layer_local_spec
from linear_moe.model.qwen2.model import GPTModel
from linear_moe.model.qwen2.transformer_config import Qwen2TransformerConfig
from linear_moe.arguments import get_patch_args
from linear_moe.tokenizer import get_tokenizer, build_tokenizer
from linear_moe.data import build_evaluation_dataset
from linear_moe.finetune_utils import build_data_loader
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from linear_moe.generation.api import generate_and_post_process
from examples.linear_moe_qwen2.pretrain_qwen import model_provider

class Qwen2LinearMoePureMamba2(BaseModel):
    def __init__(self,
                 path: str = None,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None):
        initialize_megatron(extra_args_provider=get_patch_args, ignore_unknown_args=True) # 有main则要注释该行
        args = get_args()
        # self.args = args
        self.model_provider = model_provider
        build_tokenizer(args)
        self.tokenizer = get_tokenizer()
        self.template_parser = LMTemplateParser(meta_template)
        self.max_seq_len = max_seq_len

        self.model = get_model(self.model_provider,
                    model_type=ModelType.encoder_or_decoder,
                    wrap_with_ddp=False)
        assert args.load is not None
        if args.load is not None and args.no_load_optim:
            load_checkpoint(self.model, None, None)

        torch.distributed.barrier()
        if not isinstance(self.model, list):
            self.model = [self.model]

        assert len(self.model) == 1, 'Above condition should have caught this'
        self.model = self.model[0]

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        return len(self.tokenizer.encode(prompt))

    def generate(self, inputs: List[str], max_out_len: int = 2048) -> List[str]:
        """Generate results given a list of inputs. """
        args = get_args()

        # build_tokenizer(args)

        self.model.eval()
        # print(inputs)
        num_examples = len(inputs)
        buffer = []
        outputs = []
        for idx, line in enumerate(inputs):
            line = line.strip()

            if len(buffer) < args.micro_batch_size:
                buffer.append(line)

            if len(
                    buffer
            ) == args.micro_batch_size or idx == num_examples - 1:
                # sl = args.out_seq_length
                # tk = args.top_k
                # tp = args.top_p
                sl = max_out_len
                tk = 10 # top_k
                tp = 0 # top_p
                temperature = args.temperature
                prompts_plus_generations, _, _, _ = \
                    generate_and_post_process(self.model,
                                            prompts=buffer,
                                            tokens_to_generate=sl,
                                            top_k_sampling=tk,
                                            temperature=temperature,
                                            top_p_sampling=tp)

                for prompt, p_and_g in zip(buffer,
                                            prompts_plus_generations):
                    generation = p_and_g.replace('<|endoftext|>', '')
                    print(prompt)
                    # print(p_and_g)
                    generation = generation[len(prompt):]
                    # print(generation[:max_out_len])
                    print(generation[:max_out_len + 1])
                    outputs.append(generation[:max_out_len + 1])
                buffer.clear()
        # print(len(outputs))
        return outputs

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        pass


# if __name__ == "__main__":
#     initialize_megatron(extra_args_provider=get_patch_args)
#     model = Qwen2LinearMoePureMamba2()
#     # model.generate(["Star Wars: Episode IX is scheduled for release on"])
#     model.generate(["Star Wars: Episode IX is scheduled for release on", "The construction industry partners challenging B.C.'s Community Benefits Agreement (CBA) will be"], 512)