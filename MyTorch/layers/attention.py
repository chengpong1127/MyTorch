from MyTorch import Model, Tensor
from . import Linear, Dropout
from ..activations import Softmax
import numpy as np

class Attention(Model):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = Dropout(dropout)
        self.q_linear = Linear(embed_dim, embed_dim)
        self.k_linear = Linear(embed_dim, embed_dim)
        self.v_linear = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=2)

    def forward(self, q, k, v, mask: Tensor = None):
        # q: (batch_size, seq_len, embed_dim)
        # k: (batch_size, seq_len, embed_dim)
        # v: (batch_size, seq_len, embed_dim)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        # q: (batch_size, seq_len, embed_dim)
        # k: (batch_size, seq_len, embed_dim)
        # v: (batch_size, seq_len, embed_dim)
        scaled_attention_logits = q @ k.transpose(0, 2, 1) / np.sqrt(self.embed_dim)
        # scaled_attention_logits: (batch_size, seq_len, seq_len)
        if mask != None:
            # mask: (batch_size, seq_len)
            _mask = mask.reshape(mask.shape[0], 1, -1)
            _inverse_mask = _mask * -1 + 1
            scaled_attention_logits = scaled_attention_logits + (_inverse_mask * -1e9)
        scores = self.softmax(scaled_attention_logits)
        # scores: (batch_size, seq_len, seq_len)
        scores = self.dropout(scores)
        context = scores @ v
        # context: (batch_size, seq_len, embed_dim)
        return context

class MultiheadAttention(Model):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.q_linear = Linear(embed_dim, embed_dim)
        self.k_linear = Linear(embed_dim, embed_dim)
        self.v_linear = Linear(embed_dim, embed_dim)
        self.softmax = Softmax(dim=3)

    def forward(self, q, k, v, mask: Tensor = None):
        # q: (batch_size, seq_len, embed_dim)
        # k: (batch_size, seq_len, embed_dim)
        # v: (batch_size, seq_len, embed_dim)
        batch_size = q.shape[0]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q = q.reshape(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(0, 2, 1, 3)
        # q: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        # k: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        # v: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        scaled_attention_logits = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.embed_dim / self.num_heads)
        # scaled_attention_logits: (batch_size, num_heads, seq_len, seq_len)
        if mask != None:
            # mask: (batch_size, seq_len)
            mask.requires_grad = False
            _mask = mask.reshape(mask.shape[0], 1, 1, -1)
            _inverse_mask = _mask * -1 + 1
            scaled_attention_logits = scaled_attention_logits + (_inverse_mask * -1e9)
        scores = self.softmax(scaled_attention_logits)
        # scores: (batch_size, num_heads, seq_len, seq_len)
        scores = self.dropout(scores)
        context = scores @ v
        # context: (batch_size, num_heads, seq_len, embed_dim // num_heads)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
        return context