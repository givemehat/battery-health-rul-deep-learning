import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)

def decompose_capacity_series(series: np.ndarray, window_length: int = 15, polyorder: int = 2):
    """
    Decompose capacity degradation sequence into Low-Frequency (global aging trend) 
    and High-Frequency (capacity regeneration peaks and transient sensor fluctuations) components.
    
    This implements the decomposition paradigm highlighted in Qiu et al. (2024) [CEEMDAN-TCN-LSTM],
    enabling specialized neural sub-networks to independently learn smooth thermodynamic degradation 
    and abrupt chemical relaxation dynamics without mutual gradient interference.
    """
    if len(series) < window_length:
        window_length = max(3, len(series) if len(series) % 2 != 0 else len(series) - 1)
        polyorder = min(1, window_length - 1)
        
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
        
    try:
        low_freq = savgol_filter(series, window_length=window_length, polyorder=polyorder)
    except Exception:
        low_freq = gaussian_filter1d(series, sigma=2.0)
        
    high_freq = series - low_freq
    return low_freq, high_freq

def compute_sample_entropy(time_series: np.ndarray, m: int = 2, r_factor: float = 0.2):
    """
    Compute Sample Entropy (SampEn) to assess the complexity/frequency division of decomposed modes
    as formulated in Qiu et al. (2024).
    """
    N = len(time_series)
    if N <= m + 1:
        return 0.0
    r = r_factor * np.std(time_series)
    if r == 0:
        return 0.0
        
    def _phi(m_len):
        x = np.array([time_series[i:i + m_len] for i in range(N - m_len + 1)])
        # Chebyshev distance (max coordinate diff)
        dists = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
        # Exclude self-matches
        c = (np.sum(dists < r, axis=1) - 1) / (N - m_len)
        return np.mean(c)
        
    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m > 0 and phi_m1 > 0:
            return -np.log(phi_m1 / phi_m)
        return 0.0
    except Exception:
        return 0.0
