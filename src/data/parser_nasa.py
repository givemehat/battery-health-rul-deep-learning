import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import logging

try:
    from scipy.integrate import trapezoid as trapz_fn
except ImportError:
    try:
        from numpy import trapezoid as trapz_fn
    except ImportError:
        def trapz_fn(y, x):
            return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)

logger = logging.getLogger(__name__)

EOL_THRESHOLDS = {
    'B0005': 1.40,
    'B0006': 1.40,
    'B0007': 1.50,
    'B0018': 1.40,
    'DEFAULT': 1.40
}

NOMINAL_CAPACITY = 2.0  # Ah for NASA 18650 Li-ion cells

def parse_nasa_mat_file(file_path: str):
    """
    Parse a NASA battery .mat file and extract cycle-level tabular features.
    """
    cell_id = os.path.splitext(os.path.basename(file_path))[0]
    mat = loadmat(file_path)
    
    if cell_id not in mat:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            raise ValueError(f'No valid battery key found in {file_path}')
        key = keys[0]
    else:
        key = cell_id
        
    cycle_data = mat[key][0, 0]['cycle'][0]
    
    records = []
    discharge_cycle_idx = 0
    
    for c_idx, cycle_struct in enumerate(cycle_data):
        cycle_type = cycle_struct['type'][0]
        ambient_temp = float(cycle_struct['ambient_temperature'][0, 0])
        time_arr = cycle_struct['time'][0]
        data_struct = cycle_struct['data'][0, 0]
        
        if cycle_type == 'discharge':
            discharge_cycle_idx += 1
            
            # Extract time-series sensor arrays for the cycle
            v_meas = data_struct['Voltage_measured'].flatten() if 'Voltage_measured' in data_struct.dtype.names else np.array([])
            i_meas = data_struct['Current_measured'].flatten() if 'Current_measured' in data_struct.dtype.names else np.array([])
            t_meas = data_struct['Temperature_measured'].flatten() if 'Temperature_measured' in data_struct.dtype.names else np.array([])
            t_sec = data_struct['Time'].flatten() if 'Time' in data_struct.dtype.names else np.array([])
            
            # Discharge capacity in Ah
            if 'Capacity' in data_struct.dtype.names and len(data_struct['Capacity']) > 0:
                capacity = float(data_struct['Capacity'][0, 0])
            else:
                if len(i_meas) > 1 and len(t_sec) > 1:
                    capacity = float(trapz_fn(np.abs(i_meas), t_sec) / 3600.0)
                else:
                    capacity = np.nan
                    
            # Compute statistical summary features for this discharge cycle
            v_start = float(v_meas[0]) if len(v_meas) > 0 else np.nan
            v_end = float(v_meas[-1]) if len(v_meas) > 0 else np.nan
            v_min = float(np.min(v_meas)) if len(v_meas) > 0 else np.nan
            v_mean = float(np.mean(v_meas)) if len(v_meas) > 0 else np.nan
            v_std = float(np.std(v_meas)) if len(v_meas) > 0 else np.nan
            
            i_mean = float(np.mean(np.abs(i_meas))) if len(i_meas) > 0 else np.nan
            
            t_start = float(t_meas[0]) if len(t_meas) > 0 else np.nan
            t_max = float(np.max(t_meas)) if len(t_meas) > 0 else np.nan
            t_mean = float(np.mean(t_meas)) if len(t_meas) > 0 else np.nan
            t_rise = t_max - t_start if not np.isnan(t_max) and not np.isnan(t_start) else np.nan
            
            discharge_duration = float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else np.nan
            
            # Energy discharged (Wh)
            if len(v_meas) > 1 and len(i_meas) > 1 and len(t_sec) > 1:
                energy_discharged = float(trapz_fn(v_meas * np.abs(i_meas), t_sec) / 3600.0)
            else:
                energy_discharged = np.nan
                
            records.append({
                'cell_id': cell_id,
                'cycle_index': discharge_cycle_idx,
                'raw_cycle_idx': c_idx + 1,
                'ambient_temperature': ambient_temp,
                'capacity': capacity,
                'soh': (capacity / NOMINAL_CAPACITY) * 100.0 if not np.isnan(capacity) else np.nan,
                'v_start': v_start,
                'v_end': v_end,
                'v_min': v_min,
                'v_mean': v_mean,
                'v_std': v_std,
                'i_mean': i_mean,
                't_start': t_start,
                't_max': t_max,
                't_mean': t_mean,
                't_rise': t_rise,
                'discharge_duration': discharge_duration,
                'energy_discharged': energy_discharged,
            })
            
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        eol_thresh = EOL_THRESHOLDS.get(cell_id, EOL_THRESHOLDS['DEFAULT'])
        below_thresh = df[df['capacity'] <= eol_thresh]
        if len(below_thresh) > 0:
            eol_cycle = below_thresh['cycle_index'].iloc[0]
        else:
            eol_cycle = df['cycle_index'].iloc[-1]
            
        df['eol_threshold'] = eol_thresh
        df['eol_cycle'] = eol_cycle
        df['rul_true'] = np.maximum(0, df['eol_cycle'] - df['cycle_index'])
        df['capacity_diff'] = df['capacity'].diff().fillna(0)
        df['is_regeneration'] = (df['capacity_diff'] > 0.005).astype(int)
        df['degradation_rate'] = df['capacity'].diff(5) / 5.0
        df['degradation_rate'] = df['degradation_rate'].bfill().fillna(0)
        
    return df

def load_all_nasa_cells(data_dir: str, save_processed: bool = True, output_dir: str = 'data/processed'):
    """
    Parse all available NASA battery cells in data_dir.
    """
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)
    if not mat_files:
        logger.warning(f'No .mat files found in {data_dir}')
        return pd.DataFrame()
        
    dfs = []
    target_cells = ['B0005', 'B0006', 'B0007', 'B0018']
    mat_files.sort(key=lambda p: (0 if any(c in p for c in target_cells) else 1, p))
    
    for f in mat_files:
        cell_name = os.path.splitext(os.path.basename(f))[0]
        try:
            df_cell = parse_nasa_mat_file(f)
            if len(df_cell) > 10:
                dfs.append(df_cell)
                logger.info(f'Parsed {cell_name}: {len(df_cell)} discharge cycles, EOL Cycle: {df_cell["eol_cycle"].iloc[0]}')
        except Exception as e:
            logger.error(f'Failed to parse {f}: {e}')
            
    if not dfs:
        return pd.DataFrame()
        
    all_df = pd.concat(dfs, ignore_index=True)
    
    if save_processed:
        os.makedirs(output_dir, exist_ok=True)
        all_df.to_csv(os.path.join(output_dir, 'nasa_battery_cycles.csv'), index=False)
        for cell_id, group in all_df.groupby('cell_id'):
            group.to_csv(os.path.join(output_dir, f'{cell_id}_cycles.csv'), index=False)
        logger.info(f'Saved processed datasets to {output_dir}')
        
    return all_df
