import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,features: int):
        super().__init__()

        self.alfa=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.ones(features))
        self.eps= 10**-50
        

    def forward(self,x):
        mean=torch.mean(x,dim=-1,keepdim=True) # (batch,seq,1)
        std=torch.std(x,dim=-1,keepdim=True) # (batch,seq,1)
        #(batch,seq,emb) --> (batch,seq,emb) 
        print(self.alfa.shape)
        x= (self.alfa*(x - mean)/std) + self.beta

        return x

if __name__=='__main__':
    lm=LayerNormalization(3)

    x=torch.rand(1,2,3)
    print(x)
    print(lm(x))