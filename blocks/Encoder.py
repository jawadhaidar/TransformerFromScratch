import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization
#Encoder Block 
class EncoderBlock(nn.Module):
    def __init__(self,multiHead: MultiHeadAttention ,feedForward: FeedForward,features: int,dropout: float):
        super().__init__()
        self.multiHead=multiHead
        self.feedForward=feedForward
        self.resdualConnections=nn.ModuleList([ResidualConnection(features,dropout),ResidualConnection(features,dropout)])

    def forward(self,x,src_mask):
        x=self.resdualConnections[0](x,lambda x:self.multiHead(x,x,x,src_mask))
        x=self.resdualConnections[1](x,self.feedForward)
        return x
          
#Encoder 
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
