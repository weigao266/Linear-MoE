# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from einops import rearrange


def naive_mixattention_op(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    a: torch.Tensor, # b h a_num d
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    
    softmax = torch.nn.Softmax(dim=-1)
    a_attn = softmax((a * scale) @ k.transpose(-2, -1)) # b, h, a_num, n
    a_v = a_attn @ v # b, h, a_num, d
    
    q_attn = softmax((q * scale) @ a.transpose(-2, -1)) # b, h, n, a_num
    o = q_attn @ a_v # b, h, n, d
    
    return o
