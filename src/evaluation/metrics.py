import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard prognostic evaluation metrics required by problem statement:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of Determination
    - Max Error: Worst-case absolute cycle error
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Safe MAPE avoiding division by zero
    non_zero = (y_true > 0)
    if np.sum(non_zero) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0
        
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0
    max_err = float(np.max(np.abs(y_true - y_pred)))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R2': r2,
        'Max Error': max_err
    }
