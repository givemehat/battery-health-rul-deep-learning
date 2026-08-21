import os
import glob
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

CALCE_EOL_THRESHOLD = 0.77  # Ah for CALCE CS2 series (70% of 1.1 Ah nominal)
CALCE_NOMINAL_CAPACITY = 1.1  # Ah

def parse_calce_file(file_path: str):
    """
    Parse CALCE battery cycle data from CSV or Excel file.
    """
    cell_id = os.path.splitext(os.path.basename(file_path))[0]
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df_raw = pd.read_excel(file_path)
    else:
        df_raw = pd.read_csv(file_path)
        
    # Standardize column names
    col_map = {
        'Cycle_Index': 'cycle_index',
        'Cycle': 'cycle_index',
        'Discharge_Capacity(Ah)': 'capacity',
        'Discharge_Capacity': 'capacity',
        'Capacity': 'capacity',
        'Capacity(Ah)': 'capacity'
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    
    if 'cycle_index' not in df_raw.columns:
        df_raw['cycle_index'] = np.arange(1, len(df_raw) + 1)
        
    df = pd.DataFrame({
        'cell_id': cell_id,
        'cycle_index': df_raw['cycle_index'],
        'capacity': df_raw['capacity'],
        'soh': (df_raw['capacity'] / CALCE_NOMINAL_CAPACITY) * 100.0,
        'eol_threshold': CALCE_EOL_THRESHOLD
    })
    
    # Calculate RUL
    below_thresh = df[df['capacity'] <= CALCE_EOL_THRESHOLD]
    if len(below_thresh) > 0:
        eol_cycle = below_thresh['cycle_index'].iloc[0]
    else:
        eol_cycle = df['cycle_index'].iloc[-1]
        
    df['eol_cycle'] = eol_cycle
    df['rul_true'] = np.maximum(0, df['eol_cycle'] - df['cycle_index'])
    df['capacity_diff'] = df['capacity'].diff().fillna(0)
    df['is_regeneration'] = (df['capacity_diff'] > 0.003).astype(int)
    
    return df
