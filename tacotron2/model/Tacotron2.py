"""
Title:
    An implementation of End-to-End TTS model Tacotron2 with PyTorch.
    
Description:
    This is a source code of Tacotron2 model architecture.

Author: SeungHyun Lee(@whsqkaak)
Date: 2022-12-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from torch import Tensor

from tacotron2.modules.layers import BatchNormConv1d


class Encoder(nn.Module):
    """
    An implementation of Tacotron2 Encoder.
    
    Encoder is composed of:
        - 1d Convolution layers (default 3 layers) + Batch Normalization
        - Bidirectional LSTM
        
    Args:
        enc_dim:
            Dimension of encoder embeddings.(== A number of input channels)
        num_conv_layers:
            A number of 1d convolution layers.
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
        enc_dim: int = 512,
        num_conv_layers: int = 3,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Define 1d Conv layers
        self.conv1d_bank = nn.ModuleList(
            [
                BatchNormConv1d(enc_dim, enc_dim,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=int((kernel_size - 1) / 2),
                                dilation=dilation,
                                activation=activation
                ) 
                for _ in range(num_conv_layers)
            ]
        )
        
        # Define Bidirectional LSTM
        self.lstm = nn.LSTM(enc_dim, int(enc_dim / 2),
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x: Tensor):
        """
        Calculate forward propagation of Encoder.
        
        Args:
            x:
                Batch of the input sequences.
        
        Returns:
            Batch of the output sequences.
            
        Shape:
            x: `(B, T, enc_dim)`
            Returns: `(B, T, enc_dim)`
            
            where
                B is a batch size.
                T is a time steps.
                enc_dim is a dimension of encoder embeddings.
        """
        # Transpose for convoltion
        # `(B, T, enc_dim)` -> `(B, enc_dim, T)`
        x = x.transpose(1, 2)
        
        # Apply conv1d layers with batch norm
        # `(B, enc_dim, T)` -> `(B, enc_dim, T)`
        for batchnorm_conv1d in self.conv1d_bank:
            x = self.dropout(batchnorm_conv1d(x))
        
        # Recover original shape
        # `(B, enc_dim, T)` -> `(B, T, enc_dim)`
        x = x.transpose(1, 2)
        
        # Apply LSTM
        # `(B, T, enc_dim)` -> `(B, T, enc_dim)`
        outputs, _ = self.lstm(x)
        
        return outputs
        
            
        
        
        


class Tacotron2(nn.Module):
    """
    An implementation of Tacotron2 model.
    
    Tacotron2 is composed of:
        - Embedding
        - Encoder
        - Decoder
        - Post Network
        
    Args:
        num_embed:
            Size of the dictionary of embeddings. (== length of symbols)
        embed_dim:
            The size of each embedding vector.
    """
    
    def __init__(
        self,
        num_embed: int,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embed, embed_dim)
        
        # TODO: Implement
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = PostNet()