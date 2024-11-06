import numbers
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        input_norm = (input - mean) / std
        layer_norm = self.weight * input_norm

        if self.bias is not None:
            layer_norm = layer_norm + self.bias

        return layer_norm


class PreLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], func) -> None:
        super(PreLayerNorm, self).__init__()
        self.layer_norm = LayerNorm(normalized_shape)
        self.func = func
    
    def forward(self, input: Tensor, *args: Optional[Tensor], **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        normalized_input = self.layer_norm(input)
        
        tensor_args = [arg if arg is not None else None for arg in args]
        
        result = self.func(normalized_input, *tensor_args, **kwargs)
        
        if isinstance(result, Tensor):
            return result + input
        elif isinstance(result, tuple) and isinstance(result[0], Tensor):
            return (result[0] + input,) + result[1:]
        else:
            raise ValueError("The function must return a Tensor or a tuple with a Tensor as its first element")


class PostLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], func) -> None:
        super(PostLayerNorm, self).__init__()
        self.layer_norm = LayerNorm(normalized_shape)
        self.func = func
    
    def forward(self, input: Tensor, *args: Optional[Tensor], **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        tensor_args = [arg if arg is not None else None for arg in args]

        result = self.func(input, *tensor_args, **kwargs)
        
        if isinstance(result, Tensor):
            return self.layer_norm(result + input)
        elif isinstance(result, tuple) and isinstance(result[0], Tensor):
            return (self.layer_norm(result[0] + input),) + result[1:]
        else:
            raise ValueError("The function must return a Tensor or a tuple with a Tensor as its first element")