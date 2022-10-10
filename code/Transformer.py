import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
class PreTransformer(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.head_conv = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(hidden_size, 4, 2*hidden_size, 0.1)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):

        s, b, h = _x.shape  # x is input, size (seq_len, batch, hidden_size)
        _x = _x.view(s*b, h)
        x = self.head_conv(_x)
        x = x.view(s, b, -1)
        x = self.transformer(x)

        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x