# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from einops import rearrange

def generate_a(
    x: torch.Tensor,
    y: torch.Tensor,
    a_pooling:bool,
    pooling_on_d:bool,
    a_pool: torch.nn.Module,
    a_proj: torch.nn.Module,
):
    b, h, n, d = y.shape
    if a_pooling:
        if pooling_on_d:
            y = rearrange(y, 'b h n d -> b (h n) d')
            a_y = a_pool(y)
            a_y = rearrange(a_y, 'b (h n) a_num -> b h n a_num', h=h, n=n)
        else:
            y = rearrange(y, 'b h n d -> b n (h d)')
            a_y = a_pool(y.transpose(-2, -1)).transpose(-2, -1)
            a_y = rearrange(a_y, 'b a_num (h d) -> b h a_num d', h=h, d=d)
        return a_y
    else:
        assert pooling_on_d == True
        # import pdb
        # pdb.set_trace()
        x = x.transpose(0, 1)
        a_x = a_proj(x)
        return a_x

def naive_mixattention_op(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    x: torch.Tensor, # b, h, n, hidden_size
    a_pooling:bool,
    a_pool: torch.nn.Module,
    a_proj: torch.nn.Module,
    scale: Optional[float] = None,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # q = q * scale

    # b, h, n, d = q.shape
    # q = rearrange(q, 'b h n d -> b n (h d)')
    # k = rearrange(k, 'b h n d -> b n (h d)')
    # a_q = a_proj(q.transpose(-2, -1)).transpose(-2, -1)
    # a_q = rearrange(a_q, 'b a_num (h d) -> b h a_num d', h=h, d=d)

    # a_k = a_proj(k.transpose(-2, -1)).transpose(-2, -1)
    # a_k = rearrange(a_k, 'b a_num (h d) -> b h a_num d', h=h, d=d)

    # q = rearrange(q, 'b n (h d) -> b h n d', h=h, d=d)
    # k = rearrange(k, 'b n (h d) -> b h n d', h=h, d=d)

    a_q = generate_a(x, q, a_pooling, False, a_pool, a_proj)
    a_k = generate_a(x, k, a_pooling, False, a_pool, a_proj)
    
    softmax = torch.nn.Softmax(dim=-1)
    a_attn = softmax((a_q * scale) @ k.transpose(-2, -1)) # b, h, a_num, n
    v_a = a_attn @ v # b, h, a_num, d
    
    q_attn = softmax((q * scale) @ a_k.transpose(-2, -1)) # b, h, n, a_num
    o = q_attn @ v_a # b, h, n, d
    
    return o


def naive_mixattention_op_A(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    x: torch.Tensor, # b, h, n, hidden_size
    a_pooling:bool,
    a_pool: torch.nn.Module,
    a_proj: torch.nn.Module,
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    a_q = generate_a(x, q, a_pooling, False, a_pool, a_proj)
    a_k = generate_a(x, k, a_pooling, False, a_pool, a_proj)
    
    softmax = torch.nn.Softmax(dim=-1)
    a_attn = softmax((a_q * scale) @ k.transpose(-2, -1)) # b, h, a_num, n
    v_a = a_attn @ v # b, h, a_num, d
    
    q_attn = softmax((q * scale) @ a_k.transpose(-2, -1)) # b, h, n, a_num
    o = q_attn @ v_a # b, h, n, d
    
    return o

def naive_mixattention_op_B(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    x: torch.Tensor, # b, h, n, hidden_size
    a_pooling:bool,
    a_pool: torch.nn.Module,
    a_proj: torch.nn.Module,
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    a_k = generate_a(x, k, a_pooling, True, a_pool, a_proj)
    a_v = generate_a(x, v, a_pooling, True, a_pool, a_proj)
    
    softmax = torch.nn.Softmax(dim=-1)

    k_a = k.transpose(-2, -1) @ a_v # b, h, d, a_num
    q_attn = softmax((q * scale) @ k_a) # b, h, n, a_num

    v_a = a_k.transpose(-2, -1) @ v # b, h, a_num, d
    o = q_attn @ v_a # b, h, n, d
    
    return o

def naive_mixattention_op_C(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    x: torch.Tensor, # b, h, n, hidden_size
    a_pooling:bool,
    a_pool: torch.nn.Module,
    a_proj: Optional[torch.nn.Module],
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    a_k = generate_a(x, k, a_pooling, True, a_pool, a_proj)
    a_v = generate_a(x, v, a_pooling, True, a_pool, a_proj)

    k_a = k.transpose(-2, -1) @ a_v # b, h, d, a_num
    v_a = a_k.transpose(-2, -1) @ v # b, h, a_num, d

    kv = k_a @ v_a # b, h, d, d
    o = q @ kv # b, h, n, d
    
    return o

def naive_mixattention_op_D(
    q: torch.Tensor, # b, h, n, d
    k: torch.Tensor, # b, h, n, d
    v: torch.Tensor, # b, h, n, d
    x: torch.Tensor, # b, h, n, hidden_size
    a_pooling:bool,
    a_pool: torch.nn.Module,
    a_proj: torch.nn.Module,
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    a_q = generate_a(x, q, a_pooling, False, a_pool, a_proj)
    a_k = generate_a(x, k, a_pooling, False, a_pool, a_proj)
    
    softmax = torch.nn.Softmax(dim=-1)
    
    q_attn = softmax((q * scale) @ a_k.transpose(-2, -1)) # b, h, n, a_num
    k_attn = softmax((a_q * scale) @ k.transpose(-2, -1)) # b, h, a_num, n
    qk_attn = softmax((q_attn * scale) @ k_attn) # b, h, n, n
    o = qk_attn @ v # b, h, n, d

    return o