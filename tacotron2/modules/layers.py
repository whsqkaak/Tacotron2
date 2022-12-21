"""
Title:
    An implementation of End-to-End TTS model Tacotron2 with PyTorch.
    
Description:
    This is a source code of layers of Tacotron2 model.

Author: SeungHyun Lee(@whsqkaak)
Date: 2022-12-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from torch import Tensor


class BatchNormConv1d(nn.Module):
    """
    An implementation of
    1d Convolution layer with batch normalization.
    
    Args:
        in_channels:
            A number of input channels.
        out_channels:
            A number of output channels.
        kernel_size:
            Size of the convolving kernel.
        stride:
            Stride of the convolution.
        padding:
            Padding added to both sides of the input sequences.
        dilation:
            The space between the kernel points.
        bias:
            If `True`, adds a learnable bias to the output.
        activation:
            Activation function to apply.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        
        self.activation = activation
        self.batch_norm1d = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate forward propagation.
        
        Args:
            x:
                Batch of the input sequences.
            
        Returns:
            Batch of the output sequences.
        
        Shape:
            x: `(B, T, in_channels)`
            Returns: `(B, T, out_channels)`
            
            where
                B is a batch size.
                T is a time steps.
                in_channels is a number of input channels.
                out_channels is a number of output channels.
        """
        x = self.conv1d(x)
        
        if self.activation is not None:
            x = self.activation(x)
            
        return self.batch_norm1d(x)