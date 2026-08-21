import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class BatterySequenceDataset(Dataset):
    """
    PyTorch Dataset for Battery RUL / SOH sequential temporal learning.
    """
    def __init__(self, X: np.ndarray, y_rul: np.ndarray, y_cap: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32).unsqueeze(-1)
        self.y_cap = torch.tensor(y_cap, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_cap[idx]

def create_sliding_windows(
    df: pd.DataFrame, 
    seq_len: int = 15, 
    feature_cols: Optional[List[str]] = None,
    target_col_rul: str = 'rul_true',
    target_col_cap: str = 'capacity',
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Construct temporal sliding window sequences [N, seq_len, num_features] 
    preventing any future data leakage.
    """
    if feature_cols is None:
        feature_cols = [
            'capacity', 'soh', 'v_mean', 'v_std', 't_mean', 
            'discharge_duration', 'energy_discharged', 'capacity_diff'
        ]
        
    # Keep only available cols
    avail_cols = [c for c in feature_cols if c in df.columns]
    
    # Fill missing values
    df_clean = df[avail_cols].ffill().bfill().fillna(0)
    
    if fit_scaler or scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm_data = scaler.fit_transform(df_clean.values)
    else:
        norm_data = scaler.transform(df_clean.values)
        
    rul_vals = df[target_col_rul].values if target_col_rul in df.columns else np.zeros(len(df))
    cap_vals = df[target_col_cap].values if target_col_cap in df.columns else np.zeros(len(df))
    
    X_seq, y_rul_seq, y_cap_seq = [], [], []
    
    for i in range(len(df) - seq_len + 1):
        X_seq.append(norm_data[i:i + seq_len])
        # Target is the value at the end of the sequence window
        y_rul_seq.append(rul_vals[i + seq_len - 1])
        y_cap_seq.append(cap_vals[i + seq_len - 1])
        
    return np.array(X_seq), np.array(y_rul_seq), np.array(y_cap_seq), scaler
