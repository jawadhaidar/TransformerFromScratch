import torch 
from LayerNormalization import LayerNormalization
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self,features: int, dropout: float):
        super().__init__()
        self.LayerNorm=LayerNormalization(features)
        self.dropout=nn.Dropout(dropout)
       

    def forward(self,x,Layer):
        x= x + self.dropout(Layer(self.LayerNorm(x)))
        return x