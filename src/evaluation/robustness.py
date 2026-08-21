import numpy as np
import pandas as pd
import torch
from .metrics import compute_all_metrics
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def evaluate_degradation_stages(
    model, 
    X_test: np.ndarray, 
    y_true: np.ndarray, 
    cycles: np.ndarray,
    device: torch.device,
    stages: Dict[str, Tuple[int, int]] = {
        'Early-Stage (Cycles 1-60)': (1, 60),
        'Mid-Stage (Cycles 61-120)': (61, 120),
        'Late-Stage / Near-EOL (Cycles > 120)': (121, 9999)
    }
) -> pd.DataFrame:
    """
    Degradation-stage robustness analysis required by Problem Statement.
    Evaluates prediction accuracy across Early, Mid, and Late operating life stages.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_rul, pred_cap = model(x_tensor)
        y_pred = pred_rul.cpu().numpy().flatten()
        
    results = []
    for stage_name, (c_min, c_max) in stages.items():
        mask = (cycles >= c_min) & (cycles <= c_max)
        if np.sum(mask) > 0:
            m = compute_all_metrics(y_true[mask], y_pred[mask])
            m['Stage'] = stage_name
            m['Samples'] = int(np.sum(mask))
            results.append(m)
            
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        df_res = df_res[['Stage', 'Samples', 'MAE', 'RMSE', 'MAPE (%)', 'R2', 'Max Error']]
    return df_res
