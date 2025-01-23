import torch
import torch.nn as nn 

class FeedForward(nn.Module):
    '''
    feed forward layer in transformers
    '''
    def __init__(self,features: int,hidden_state: int, dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(features,hidden_state,bias=True)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(hidden_state,features,bias=True)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.linear2(self.dropout(self.relu(self.linear1(x))))
        return x

if __name__=="__main__":
    # Example usage
    d_model = 512  # Model dimension
    d_ff = 2048    # Hidden dimension
    feedforward = FeedForward(d_model, d_ff)

    # Sample input (batch_size=2, seq_length=10, d_model=512)
    x = torch.randn(2, 10, d_model)
    output = feedforward(x)
    print(output.shape)  # Output shape: (2, 10, 512)