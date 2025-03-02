import torch
import torch.nn as nn
import math

# Define an embedding module
class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        """
        Initialize the embedding layer.

        Args:
            vocab_size (int): The number of unique words (or tokens) in the vocabulary.
            emb_size (int): The dimensionality of the embedding vectors.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        # Create an embedding layer that maps input indices to dense vectors
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
    
    def forward(self, x):
        """
        Forward pass for the embedding layer.

        Args:
            x (Tensor): A tensor containing token indices.

        Returns:
            Tensor: The corresponding embeddings, scaled by the square root of emb_size.
        """
        return self.emb(x) * math.sqrt(self.emb_size)

# Test the embedding module
if __name__ == "__main__":
    emb = Embedding(10, 4)  # Create an embedding layer with vocab size 10 and embedding dimension 4
    x = torch.tensor([1, 2, 3, 4])  # Define a sample input tensor with token indices
    print(emb(x))  # Print the output embeddings
