import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Standard Multi-Layer LSTM for Battery Remaining Useful Life and SOH Forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_rul = nn.Linear(hidden_dim, 1)
        self.fc_cap = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        out, (hn, cn) = self.lstm(x)
        last_step = self.dropout(out[:, -1, :])
        pred_rul = self.fc_rul(last_step)
        pred_cap = self.fc_cap(last_step)
        return pred_rul, pred_cap

class GRUModel(nn.Module):
    """
    Standard Multi-Layer GRU for Battery Health & RUL Prognostics.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_rul = nn.Linear(hidden_dim, 1)
        self.fc_cap = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, hn = self.gru(x)
        last_step = self.dropout(out[:, -1, :])
        pred_rul = self.fc_rul(last_step)
        pred_cap = self.fc_cap(last_step)
        return pred_rul, pred_cap
