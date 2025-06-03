import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    """Embedd text seq and apply positioanl encoding"""
    def __init__(self, vocab_size, batch_size, d_embed=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed

        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.scalar = 10000.

    def postional_encoding(self,x):
        """
        Apply positional encoding to the input tensor.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_embed)
        Returns:
            Tensor with positional encoding applied.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        seq_idx = torch.arange(0, seq_len, device=device).unsqueeze(1).expand(seq_len, self.d_embed)
        emedding_idx = torch.arange(0, self.d_embed, device=device).unsqueeze(0).expand(seq_len, self.d_embed)
        pos = seq_idx / torch.pow(self.scalar, (2 * emedding_idx) / self.d_embed)

        # Apply sine and cosine functions to even and odd indices
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])


        # add batch dimension and copy embedding for each batch
        return pos.to(device).unsqueeze(0).expand(batch_size, seq_len, self.d_embed)

    def forward(self, x):
        """
        Forward pass of the embedding layer.
        Args:
            x: Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, d_embed) with positional encoding applied.
        """
        x = self.embedding(x).to(device)
        return x + self.postional_encoding(x)


class SelfAttention(nn.Module):
    def __init__(self, d_embed, d_attn, decoding=False):
        super().__init__()
        self.d_embed = d_embed
        self.d_attn = d_attn
        self.decoding = decoding

        self.W_q = nn.Linear(d_embed, d_attn)
        self.W_k = nn.Linear(d_embed, d_attn)
        self.W_v = nn.Linear(d_embed, d_attn)
        self.softmax = nn.Softmax(dim=2) # Softmax over the attention scores

        self.scalar = 1.0 / (d_attn ** 0.5)  # Scaling factor for attention scores

    def forward(self, x):
        """
        Forward pass of the self-attention layer.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_embed)
        Returns:
            Tensor of shape (batch_size, seq_len, d_attn) after applying self-attention.
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute attention scores softmax(Q @ K^T/ self.scalar) @ V
        attn_scores = torch.matmul(Q, torch.transpose(K, 1, 2)) * self.scalar
        print("attn_scores: ", attn_scores.shape)
        if self.decoding: # If decoding, mask future tokens
            mask = torch.triu(torch.ones(attn_scores.shape[-2:], device=device), diagonal=1).bool()
            attn_scores.masked_fill_(mask, float('-inf'))

        attn_weights = self.softmax(attn_scores)

        return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_embedd,  decoding=False, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0 print(f"d_model {d_model} must be divisible by n_heads: {n_heads}")

        self.d_attn = d_model // n_heads
        self.d_model = d_model
        self.attenion_heads = nn.ModuleList([SelfAttention(d_embed, self.d_attn, decoding) \
                                            for h in range(n_heads)])
        # Output linear layer to combine the attention heads
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        out_zs = []
        # Apply each attention head to the input
        for h in self.attenion_heads:
            out_zs.append(h(x))

        # Concat horizontally and apply output linear layer
        return self.output_linear(nn.cat(out_zs, -1))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """Simple 2 layer FF model.
        @input: d_model input 
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or nn.GELU()
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # x: [batch, seq_len, d_model]

if __name__ == '__main__':
    vocab_size = 1000
    d_embed = 512
    seq_len = 10
    batch_size = 32

    model = Embedding(vocab_size, d_embed)
    input_tensor = torch.randint(0, vocab_size,size=(batch_size, seq_len )).to(device)
    print(input_tensor.shape)  # Should be (seq_len, )

    embedded_tensor = model(input_tensor)
    print(embedded_tensor.shape)  # Should be (batch_size, seq_len, d_embed)
    attn = SelfAttention(d_embed, d_embed // 8)
    attn_output = attn(embedded_tensor)
    print(attn_output.shape)  # Should be (batch_size, seq_len, d_embed // 8)
