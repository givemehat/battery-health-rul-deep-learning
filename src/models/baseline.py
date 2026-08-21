import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def double_exponential(k, a, b, c, d):
    """Standard electrochemical empirical capacity degradation model: C(k) = a*exp(b*k) + c*exp(d*k)"""
    return a * np.exp(b * k) + c * np.exp(d * k)

class EmpiricalDegradationPrognosticator:
    """
    Simple prognostic baseline fitting physical double-exponential degradation trajectory.
    """
    def __init__(self, eol_threshold: float = 1.40):
        self.eol_threshold = eol_threshold
        self.popt = None
        
    def fit(self, cycles: np.ndarray, capacities: np.ndarray):
        p0 = [capacities[0], -0.001, 0.05, -0.01]
        try:
            popt, _ = curve_fit(double_exponential, cycles, capacities, p0=p0, maxfev=10000, bounds=([0, -0.1, 0, -0.1], [3.0, 0.0, 1.0, 0.0]))
            self.popt = popt
        except Exception:
            # Fallback to linear regression
            poly = np.polyfit(cycles, capacities, 1)
            self.popt = ('linear', poly)
            
    def predict_capacity(self, future_cycles: np.ndarray) -> np.ndarray:
        if self.popt is None:
            raise ValueError('Model is not fitted yet.')
        if isinstance(self.popt, tuple) and self.popt[0] == 'linear':
            return np.polyval(self.popt[1], future_cycles)
        return double_exponential(future_cycles, *self.popt)
        
    def predict_rul(self, current_cycle: int, max_search: int = 500) -> int:
        test_cycles = np.arange(current_cycle, current_cycle + max_search)
        pred_caps = self.predict_capacity(test_cycles)
        below_eol = np.where(pred_caps <= self.eol_threshold)[0]
        if len(below_eol) > 0:
            return int(below_eol[0])
        return max_search

class ClassicalMLBaseline:
    """
    Random Forest and Support Vector Regression baselines on sliding feature vectors.
    """
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        if model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svr':
            self.model = SVR(kernel='rbf', C=10.0, epsilon=0.01)
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Flatten sequence: [N, seq_len * features]
        N = X.shape[0]
        X_flat = X.reshape(N, -1)
        self.model.fit(X_flat, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X_flat = X.reshape(N, -1)
        return self.model.predict(X_flat)
