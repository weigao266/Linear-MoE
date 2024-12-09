from dataclasses import dataclass

import torch
from typing import Optional
from einops import rearrange
from megatron.core.transformer.module import MegatronModule
from transformers.activations import ACT2FN
import torch.nn.functional as F

from linear_moe.model.common_modules import RMSNorm
from fla.ops.retention import (chunk_retention, fused_chunk_retention,
                               fused_recurrent_retention, parallel_retention)


class Retention(MegatronModule):

    def __init__(
        self, 
        config,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
    ):
        super().__init__(config)
        
        self.la_mode = config.la_mode
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # num_kv_heads here mains num_query_groups
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.la_feature_map = config.la_feature_map
        if self.la_feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.la_feature_map_fn = elu
        else:
            self.la_feature_map_fn = ACT2FN[self.la_feature_map] if self.la_feature_map is not None else None

        self.key_dim = int(config.hidden_size * expand_k)
        self.value_dim = int(config.hidden_size * expand_v)

        assert self.la_mode in ['chunk', 'fused_chunk', 'parallel', 'fused_recurrent'], f"Not supported mode `{self.la_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"
        
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        if config.la_output_norm == 'rmsnorm':
            self.la_output_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=config.la_elementwise_affine, eps=config.la_norm_eps)
        elif config.la_output_norm == 'identity':
            self.la_output_norm = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{self.la_output_norm}`.")
        
        if self.la_mode == 'chunk':
            self._la_impl = chunk_retention
        elif self.la_mode == 'fused_chunk':
            self._la_impl = fused_chunk_retention
        elif self.la_mode == 'fused_recurrent':
            self._la_impl = fused_recurrent_retention
        elif self.la_mode == 'parallel':
            self._la_impl = parallel_retention
        
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: torch.nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        module._is_hf_initialized = True


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # torch.Size([128, 4, 16, 32])
        q, k, v = (rearrange(x, 'n b h d -> b h n d') for x in (q, k, v))

        if self.la_feature_map_fn is not None:
            q, k = map(self.la_feature_map_fn, (q, k))

        # expects q: B, H, T, K
        output, _ = self._la_impl(q, k, v)
        output = self.la_output_norm(output)

        output = rearrange(output, 'b h n d -> n b (h d)')

        return output
