import torch
import torch.nn as nn
from LayerNormalization import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self,features: int, dropout: float):
        super().__init__()
        self.LayerNorm=LayerNormalization(features)
        self.dropout=nn.Dropout(dropout)
       

    def forward(self,x,Layer):
        x= x + self.dropout(Layer(self.LayerNorm(x)))
        return x 

if __name__=='__main__':
    subLayer=nn.Linear(3,10)
    residual=ResidualConnection(3,0.2)
    inp=torch.rand(1,5,3)

    out=residual(inp,subLayer)
    print(out.shape)



