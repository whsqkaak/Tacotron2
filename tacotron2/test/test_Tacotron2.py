"""
Title:
    A test code of `tacotron2/model/Tacotron2.py`
    
Description:
    This is a test code of Tacotron2 model using pytest.
    
Usage:
    pytest tacotron2/test/test_Tacotron2.py
    
Author: SeungHyun Lee(@whsqkaak)
Date: 2022-12-21
"""

import pytest
import torch

from tacotron2.model.Tacotron2 import Encoder


def encoder_test():
    batch_size = 3
    time_steps = 8
    enc_dim = 10
    
    inputs = torch.randn(batch_size, time_steps, enc_dim)
    
    encoder = Encoder(enc_dim)
    
    outputs = encoder(inputs)
    
    assert outputs.shape == (batch_size, time_steps, enc_dim)
    
    
def test_main():
    """
    pytest will run this main function.
    """
    encoder_test()