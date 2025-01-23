import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.emb=nn.Embedding(self.vocab_size,self.emb_size)
    
    def forward(self,x):
        return self.emb(x)*math.sqrt(self.emb_size)
    
 
if __name__ == "__main__":
    emb=Embedding(10,4)
    x=torch.tensor([1,2,3,4])
    print(emb(x))
    



