import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    """Embedd text seq and apply positioanl encoding"""
    def __init__(self, vocab_size, d_embed=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed

        self.embedding = nn.Embedding(se


