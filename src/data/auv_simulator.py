import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_auv_mission_profile(num_cycles: int = 200, seed: int = 42):
    """
    Simulate realistic AUV (Autonomous Underwater Vehicle) battery mission profiles:
    - Dynamic pulsed thruster power surges during deep diving and ocean current counter-thrust.
    - Cold marine temperature variations (4°C deep bathymetric ocean floor to 25°C surface recovery).
    - Capacity regeneration during long dock charging / idle surface floating intervals.
    - Non-linear electrochemical fade dynamics matching marine Li-ion pack aging.
    """
    np.random.seed(seed)
    cycles = np.arange(1, num_cycles + 1)
    
    # Base nominal capacity 2.0 Ah (standard 18650 marine pack cell)
    nominal_cap = 2.0
    eol_thresh = 1.40
    
    # Electrochemical aging model with double exponential decay: C(k) = a*exp(b*k) + c*exp(d*k)
    base_fade = 2.02 * np.exp(-0.0018 * cycles) + 0.03 * np.exp(-0.012 * cycles)
    
    # Add capacity regeneration after rest intervals (every 15-25 cycles)
    regeneration = np.zeros(num_cycles)
    rest_events = [20, 42, 68, 95, 120, 150, 178]
    for r in rest_events:
        if r < num_cycles:
            # Rebound pulse with decaying memory
            decay_len = min(8, num_cycles - r)
            pulse = 0.025 * np.exp(-0.35 * np.arange(decay_len))
            regeneration[r:r+decay_len] += pulse
            
    # Add marine environmental thermal load & operational noise
    ocean_temp_depth = 4.0 + 16.0 * (1.0 + np.sin(2 * np.pi * cycles / 30.0)) / 2.0  # 4°C to 20°C
    temp_effect = -0.0008 * (25.0 - ocean_temp_depth)  # colder temp reduces available discharge capacity
    
    noise = np.random.normal(0, 0.004, num_cycles)
    
    capacity = base_fade + regeneration + temp_effect + noise
    capacity = np.clip(capacity, 1.1, 2.05)
    
    # Find EOL
    below_eol = np.where(capacity <= eol_thresh)[0]
    eol_cycle = below_eol[0] + 1 if len(below_eol) > 0 else num_cycles
    
    rul_true = np.maximum(0, eol_cycle - cycles)
    
    df = pd.DataFrame({
        'cell_id': 'AUV_PACK_CELL_01',
        'cycle_index': cycles,
        'capacity': capacity,
        'soh': (capacity / nominal_cap) * 100.0,
        'ambient_temperature': ocean_temp_depth,
        'v_mean': 3.65 - 0.002 * cycles + np.random.normal(0, 0.01, num_cycles),
        't_mean': ocean_temp_depth + 8.5 + np.random.normal(0, 0.5, num_cycles),
        'discharge_duration': 3200 - 6.5 * cycles + np.random.normal(0, 15, num_cycles),
        'energy_discharged': (3.65 - 0.002 * cycles) * capacity * 3.6,
        'eol_threshold': eol_thresh,
        'eol_cycle': eol_cycle,
        'rul_true': rul_true,
        'capacity_diff': pd.Series(capacity).diff().fillna(0),
        'is_regeneration': (pd.Series(capacity).diff().fillna(0) > 0.005).astype(int),
        'mission_profile': ['SURVEY_CRUISE' if c % 3 == 0 else ('DEEP_DIVE' if c % 3 == 1 else 'SURFACE_TRANSIT') for c in cycles]
    })
    
    return df
