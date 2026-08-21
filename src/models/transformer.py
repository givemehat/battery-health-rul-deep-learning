import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalTransformerModel(nn.Module):
    """
    Temporal Transformer model for long-range Battery RUL and SOH sequence learning.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super(TemporalTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_rul = nn.Linear(d_model, 1)
        self.fc_cap = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x: [B, T, input_dim]
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h_enc = self.transformer_encoder(h)
        last_step = self.dropout(h_enc[:, -1, :])
        
        pred_rul = self.fc_rul(last_step)
        pred_cap = self.fc_cap(last_step)
        return pred_rul, pred_cap
