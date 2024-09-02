from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
from megatron.core.transformer.module import MegatronModule


class GLAGate(MegatronModule):

    def __init__(self, config):
        super().__init__(config)
        
        self.hidden_size = config.hidden_size
        self.key_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        # num_kv_heads here mains num_query_groups
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.gla_la_gate_low_rank_dim = config.gla_la_gate_low_rank_dim
        self.gk_proj = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.gla_la_gate_low_rank_dim, bias=False),
            torch.nn.Linear(self.gla_la_gate_low_rank_dim, self.key_dim_per_group, bias=True))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        gk = self.gk_proj(hidden_states)

        return gk
