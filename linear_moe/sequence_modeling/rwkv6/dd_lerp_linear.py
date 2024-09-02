from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from megatron.core.transformer.module import MegatronModule


class LerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


class DDLerpLinear(MegatronModule):

    def __init__(
        self,
        config,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__(config)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, mu: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * mu)


class LoRA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim, bias=bias)
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)