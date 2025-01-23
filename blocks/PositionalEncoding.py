import torch
import numpy as np 
import math 

class PositionalEncoding:
    '''
    Generates the positional encoding matrix
    
    
    '''
    def __init__(self,emb_size,seq_len):
        self.emb_size=emb_size
        self.seq_len=seq_len
        self.positions=torch.arange(0,self.seq_len)
        print(self.positions.shape)

    def build_full_matrix(self): #first way 
        '''
        Generates the positional encoding matrix
        '''
        pe=torch.zeros(self.seq_len,self.emb_size)
        for pos in range(self.seq_len):
            for i in range(0,self.emb_size,2):
                pe[pos,i]=torch.sin(pos/(10000**(i/self.emb_size)))
                pe[pos,i+1]=torch.cos(pos/(10000**(i/self.emb_size)))
        return pe.unsqueeze(0)
    
    def positional_encoding(self):
        '''
        Generates the positional encoding matrix using PyTorch
        '''
        # Create a matrix of shape [position, d_model] where each element is the position index
        col = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1) 
        row=torch.pow(10000, (2 * (torch.arange(self.emb_size, dtype=torch.float32) // 2)) / self.emb_size)
        angle_rads=col/row
        
        # Apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        
        # Add a new dimension at the beginning
        pos_encoding = angle_rads.unsqueeze(0)
        
        return pos_encoding
    
    #umar jamil
    def umar_jamil(self):
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.emb_size)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.emb_size, 2).float() * (-math.log(10000.0) / self.emb_size)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self,x):
        x= x + self.build_full_matrix() #TODO: make sure the dimesions are correct 




if __name__ == "__main__":
    column=torch.tensor([1,1,1,1]).unsqueeze(1)
    row=torch.tensor([1,2,3]).unsqueeze(0)
    print(column,row)
    print(column*row)
    print(row*column)
    print(column/row)
    print(row/column)

