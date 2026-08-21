import torch
import torch.nn as nn
import torch.nn.functional as F
from .tcn import TemporalBlock
from .bilstm_attention import MultiHeadSelfAttention

class SqueezeExcitation(nn.Module):
    """Channel attention for multi-scale temporal convolutions."""
    def __init__(self, channels: int, reduction: int = 4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        w = torch.mean(x, dim=2) # [B, C]
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1) # [B, C, 1]
        return x * w

class HybridCEEMDANTCNBiLSTMDualAttention(nn.Module):
    """
    Proposed State-of-the-Art Hybrid Architecture:
    Combines:
    1. Dilated Temporal Convolutional Network (TCN) + Squeeze-and-Excitation Channel Attention 
       to capture high-frequency capacity regeneration anomalies and sensor transients.
    2. Bidirectional LSTM + Multi-Head Self-Attention to capture global thermodynamic 
       monotonic capacity fade trends.
    3. Gated Multimodal Cross-Fusion Network dynamically weighting local dynamics vs long-term degradation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, tcn_channels: list = [32, 64], num_heads: int = 4, dropout: float = 0.2):
        super(HybridCEEMDANTCNBiLSTMDualAttention, self).__init__()
        
        # Branch 1: High-Frequency / Local Dynamics TCN Branch
        tcn_layers = []
        for i, ch in enumerate(tcn_channels):
            in_ch = input_dim if i == 0 else tcn_channels[i - 1]
            tcn_layers.append(TemporalBlock(in_ch, ch, kernel_size=3, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        self.se_attn = SqueezeExcitation(tcn_channels[-1])
        
        # Branch 2: Low-Frequency / Global Trend Recurrent Branch
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temporal_attn = MultiHeadSelfAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
        # Fusion & Gating Layer
        fusion_dim = tcn_channels[-1] + hidden_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        self.fc_shared = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_rul = nn.Linear(128, 1)
        self.fc_cap = nn.Linear(128, 1)
        
    def forward(self, x):
        # x: [B, T, D]
        # TCN path
        x_perm = x.permute(0, 2, 1) # [B, D, T]
        tcn_feat = self.tcn(x_perm) # [B, C, T]
        tcn_feat = self.se_attn(tcn_feat)
        h_tcn = tcn_feat[:, :, -1] # [B, C]
        
        # BiLSTM path
        lstm_out, _ = self.bilstm(x) # [B, T, 2*H]
        attn_out, _ = self.temporal_attn(lstm_out)
        norm_out = self.norm(lstm_out + attn_out)
        h_lstm = torch.mean(norm_out, dim=1) # [B, 2*H]
        
        # Concatenate representations
        combined = torch.cat([h_tcn, h_lstm], dim=-1)
        gated = combined * self.gate(combined)
        
        shared = self.fc_shared(gated)
        pred_rul = self.fc_rul(shared)
        pred_cap = self.fc_cap(shared)
        return pred_rul, pred_cap
