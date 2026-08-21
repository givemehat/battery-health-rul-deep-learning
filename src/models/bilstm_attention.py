import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention over sequential time steps."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        Q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T, d]
        K = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V) # [B, H, T, d]
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context), attn_weights

class BiLSTMAttentionModel(nn.Module):
    """
    Bidirectional LSTM coupled with Multi-Head Self-Attention for battery RUL prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super(BiLSTMAttentionModel, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = MultiHeadSelfAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_rul = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.fc_cap = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [B, T, D]
        lstm_out, _ = self.bilstm(x) # [B, T, hidden_dim * 2]
        attn_out, weights = self.attention(lstm_out)
        fused = self.norm(lstm_out + attn_out)
        
        # Weighted context vector across time
        pool = torch.mean(fused, dim=1) # [B, hidden_dim * 2]
        pool = self.dropout(pool)
        
        pred_rul = self.fc_rul(pool)
        pred_cap = self.fc_cap(pool)
        return pred_rul, pred_cap
