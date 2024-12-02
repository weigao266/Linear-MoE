from dataclasses import dataclass

import torch
from einops import rearrange
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule

from linear_moe.model.common_modules.feature_map import (DPFPFeatureMap, HadamardFeatureMap,
                                     HedgehogFeatureMap, T2RFeatureMap)
from linear_moe.model.common_modules import RMSNorm
from .mixattention_op import naive_mixattention_op


class MixAttention(MegatronModule):

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
        self.key_dim = int(config.hidden_size * expand_k)
        self.value_dim = int(config.hidden_size * expand_v)
        
        self.la_feature_map = config.la_feature_map
        self.la_tie_feature_map_qk = config.la_tie_feature_map_qk
        
        self.la_norm_q = config.la_norm_q
        self.la_norm_k = config.la_norm_k
        self.la_do_feature_map_norm = config.la_do_feature_map_norm
        
        assert self.la_mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not supported mode `{self.la_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"
        
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        
        if self.la_feature_map == 'hedgehog':
            if self.la_tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif self.la_feature_map == 't2r':
            if self.la_tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif self.la_feature_map == 'elementwise_product':
            if self.la_tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif self.la_feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif self.la_feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif self.la_feature_map == 'relu':
            self.feature_map_q = torch.nn.ReLU()
            self.feature_map_k = torch.nn.ReLU()

        elif self.la_feature_map == 'identity':
            self.feature_map_q = torch.nn.Identity()
            self.feature_map_k = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{self.la_feature_map}`.")
        
        if config.la_output_norm == 'rmsnorm':
            self.la_output_norm = RMSNorm(hidden_size=self.head_v_dim)
        elif config.la_output_norm == 'identity':
            self.la_output_norm = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{self.la_output_norm}`.")
        
        self._la_impl = naive_mixattention_op
        
        self.a_num = 256
        pool_size = int(self.a_num ** 0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        
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

        # q,k,v shape: torch.Size([1024, 1, 8, 128])
        q, k, v = (rearrange(x, 'n b h d -> b h n d') for x in (q, k, v))

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.la_norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.la_norm_k:
            k = k / (k.sum(-1, True) + 1e-4)
        
        b, h, n, d = q.shape
        q = rearrange(q, 'b h n d -> b n (h d)')
        c = h * d
        w = int(n ** 0.5)
        a = self.pool(q.reshape(b, w, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1) # [b, a_num, c]
        a = rearrange(a, 'b a_num (h d) -> b h a_num d', h=self.num_heads, d=self.head_qk_dim)

        # expects q: b, h, n, d
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads, d=self.head_qk_dim)
        output = self._la_impl(q, k, v, a, normalize=self.la_do_feature_map_norm)
        output = self.la_output_norm(output)

        output = rearrange(output, 'b h n d -> n b (h d)')

        return output
