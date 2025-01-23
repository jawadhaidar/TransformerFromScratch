import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.dk = d_model  # Full model dimension
        self.heads = heads
        assert self.dk % self.heads == 0, "d_model must be divisible by heads"
        self.dv = d_model // heads  # Dimension per head

        self.w_K = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.w_Q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_V = nn.Linear(d_model, d_model, bias=False)  # Value projection

        self.out = nn.Linear(d_model, d_model)  # Final projection layer
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_scores=None

    def attention(self, x, mask):
        # Compute query, key, and value matrices
        k = self.w_K(x)  # (batch, seq_len, d_model)
        Q = self.w_Q(x)
        V = self.w_V(x)

        # Reshape into (batch, heads, seq_len, dv)
        k = k.view(k.shape[0], k.shape[1], self.heads, self.dv).permute(0, 2, 1, 3)
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.dv).permute(0, 2, 1, 3)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.dv).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        attention = torch.matmul(Q, k.transpose(-1, -2)) / math.sqrt(self.dv)  # (batch, heads, seq_len, seq_len)

        # Mask the attention
        attention.masked_fill_(mask[:, None, :, :] == 0, -1e9)  # Broadcast mask

        # Apply softmax and dropout
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        self.attention_scores=attention
        # Multiply attention weights with values
        projected = torch.matmul(attention, V)  # (batch, heads, seq_len, dv)

        # Concatenate heads and project back to d_model
        projected = projected.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, heads, dv)
        projected = projected.view(projected.shape[0], projected.shape[1], self.dk)  # (batch, seq_len, d_model)

        return self.out(projected)  # Final projection to (batch, seq_len, d_model)

    def forward(self, x, mask):
        return self.attention(x, mask)


if __name__=='__main__':
    pass