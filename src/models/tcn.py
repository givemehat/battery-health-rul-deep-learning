import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ChausalConv1d(nn.Module):
    """Causal 1D Convolution ensuring no future leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        
    def forward(self, x):
        # x: [batch, channels, seq_len]
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

class TemporalBlock(nn.Module):
    """Residual block with dilated causal convolutions, ReLU, and Dropout."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN) architecture inspired by Qiu et al. (2024).
    Uses dilated causal convolutions with exponentially growing dilation factors.
    """
    def __init__(self, input_dim: int, num_channels: list = [32, 64, 128], kernel_size: int = 3, dropout: float = 0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))
            
        self.network = nn.Sequential(*layers)
        self.fc_rul = nn.Linear(num_channels[-1], 1)
        self.fc_cap = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> permute to [batch_size, input_dim, seq_len]
        x_perm = x.permute(0, 2, 1)
        y = self.network(x_perm)
        # Take the final temporal feature map slice
        last_step = y[:, :, -1]
        pred_rul = self.fc_rul(last_step)
        pred_cap = self.fc_cap(last_step)
        return pred_rul, pred_cap
