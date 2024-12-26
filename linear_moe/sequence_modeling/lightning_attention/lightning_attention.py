import torch
from einops import rearrange
from megatron.core.transformer.module import MegatronModule
from transformers.activations import ACT2FN
from linear_moe.model.common_modules import RMSNorm
from lightning_attn.ops.triton import lightning_attn2, lightning_attn2_no_decay
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


class LightningAttention(MegatronModule):
    """Lightning Attention"""
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
        self.num_kv_heads = config.num_query_groups if config.num_query_groups is not None else config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.la_feature_map = config.la_feature_map
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
        
        # slope_rate in lightning attention
        slope_rate_heads = self.num_heads
        slope_rate = self._build_slope_tensor(slope_rate_heads)
        slope_rate = slope_rate.to(torch.cuda.current_device())
        
        if config.tensor_model_parallel_size > 1:
            self.slope_rate = torch.chunk(
                slope_rate,
                config.tensor_model_parallel_size,
                0,
            )[get_tensor_model_parallel_rank()]
        else:
            self.slope_rate = slope_rate
        
        if self.slope_rate is None:
            self._la_impl = lightning_attn2_no_decay
        else:
            self._la_impl = lightning_attn2
        
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: torch.nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(torch.math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if torch.math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )
            else:
                closest_power_of_2 = 2 ** torch.math.floor(
                    torch.math.log2(n)
                )
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][
                        : n - closest_power_of_2
                    ]
                )

        slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
            n_attention_heads, 1, 1
        )
        return slopes


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
        if self.slope_rate is None:
            output = self._la_impl(q, k, v)
        else:
            output = self._la_impl(q, k, v, self.slope_rate)
        output = self.la_output_norm(output)
        output = rearrange(output, 'b h n d -> n b (h d)')
        return output
