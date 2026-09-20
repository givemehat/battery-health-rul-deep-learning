import os
import glob
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as trapz_fn
import logging

logger = logging.getLogger(__name__)

EOL_THRESHOLDS = {
    'B0005': 1.40,
    'B0006': 1.40,
    'B0007': 1.50,
    'B0018': 1.40,
    'DEFAULT': 1.40
}

NOMINAL_CAPACITY = 2.0  # Ah

def load_from_cleaned_dataset(cleaned_dir: str = 'data/cleaned_dataset', save_processed: bool = True, output_dir: str = 'data/processed'):
    """
    Load and process cycle-by-cycle telemetry directly from cleaned_dataset:
    - Ingests metadata.csv (7,565 operations across 34 batteries)
    - Reads high-frequency time-series from data/<filename>
    - Computes summary statistics (voltage, thermal rise, duration, energy)
    - Calculates SOH, EOL failure cycles, RUL, and capacity regeneration indicators.
    """
    meta_path = os.path.join(cleaned_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found in {cleaned_dir}")
        
    meta = pd.read_csv(meta_path)
    data_dir = os.path.join(cleaned_dir, 'data')
    
    # Filter discharge cycles
    discharges = meta[meta['type'] == 'discharge'].copy()
    
    # Group by battery_id
    all_cell_dfs = []
    
    target_cells = ['B0005', 'B0006', 'B0007', 'B0018']
    all_batteries = sorted(discharges['battery_id'].unique().tolist(), 
                           key=lambda b: (0 if b in target_cells else 1, b))
    
    for b_id in all_batteries:
        cell_meta = discharges[discharges['battery_id'] == b_id].copy().reset_index(drop=True)
        if len(cell_meta) == 0:
            continue
            
        records = []
        for cycle_idx, row in cell_meta.iterrows():
            fn = row['filename']
            raw_cap = row['Capacity']
            ambient_temp = float(row['ambient_temperature']) if pd.notna(row['ambient_temperature']) else 24.0
            
            # Read sensor readings CSV
            csv_path = os.path.join(data_dir, fn)
            if os.path.exists(csv_path):
                df_seq = pd.read_csv(csv_path)
                v_meas = df_seq['Voltage_measured'].values if 'Voltage_measured' in df_seq.columns else np.array([])
                i_meas = df_seq['Current_measured'].values if 'Current_measured' in df_seq.columns else np.array([])
                t_meas = df_seq['Temperature_measured'].values if 'Temperature_measured' in df_seq.columns else np.array([])
                t_sec = df_seq['Time'].values if 'Time' in df_seq.columns else np.array([])
                
                # Capacity parsing / fallback integration
                try:
                    cap_val = float(raw_cap)
                except Exception:
                    if len(i_meas) > 1 and len(t_sec) > 1:
                        cap_val = float(trapz_fn(np.abs(i_meas), t_sec) / 3600.0)
                    else:
                        cap_val = np.nan
                        
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
                
                if len(v_meas) > 1 and len(i_meas) > 1 and len(t_sec) > 1:
                    energy_discharged = float(trapz_fn(v_meas * np.abs(i_meas), t_sec) / 3600.0)
                else:
                    energy_discharged = np.nan
            else:
                try:
                    cap_val = float(raw_cap)
                except Exception:
                    cap_val = np.nan
                v_start = v_end = v_min = v_mean = v_std = i_mean = t_start = t_max = t_mean = t_rise = discharge_duration = energy_discharged = np.nan

            records.append({
                'cell_id': b_id,
                'cycle_index': cycle_idx + 1,
                'ambient_temperature': ambient_temp,
                'capacity': cap_val,
                'soh': (cap_val / NOMINAL_CAPACITY) * 100.0 if not np.isnan(cap_val) else np.nan,
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
                'source_filename': fn
            })
            
        df_cell = pd.DataFrame(records)
        df_cell['capacity'] = df_cell['capacity'].ffill().bfill()
        df_cell['soh'] = (df_cell['capacity'] / NOMINAL_CAPACITY) * 100.0
        
        # Calculate RUL
        eol_thresh = EOL_THRESHOLDS.get(b_id, EOL_THRESHOLDS['DEFAULT'])
        below_thresh = df_cell[df_cell['capacity'] <= eol_thresh]
        if len(below_thresh) > 0:
            eol_cycle = below_thresh['cycle_index'].iloc[0]
        else:
            eol_cycle = df_cell['cycle_index'].iloc[-1]
            
        df_cell['eol_threshold'] = eol_thresh
        df_cell['eol_cycle'] = eol_cycle
        df_cell['rul_true'] = np.maximum(0, df_cell['eol_cycle'] - df_cell['cycle_index'])
        df_cell['capacity_diff'] = df_cell['capacity'].diff().fillna(0)
        df_cell['is_regeneration'] = (df_cell['capacity_diff'] > 0.005).astype(int)
        df_cell['degradation_rate'] = df_cell['capacity'].diff(5) / 5.0
        df_cell['degradation_rate'] = df_cell['degradation_rate'].bfill().fillna(0)
        
        all_cell_dfs.append(df_cell)
        logger.info(f"Loaded {b_id}: {len(df_cell)} discharge cycles, EOL at Cycle {eol_cycle}")
        
    master_df = pd.concat(all_cell_dfs, ignore_index=True)
    
    if save_processed:
        os.makedirs(output_dir, exist_ok=True)
        master_df.to_csv(os.path.join(output_dir, 'cleaned_dataset_battery_cycles.csv'), index=False)
        for cell_id, group in master_df.groupby('cell_id'):
            group.to_csv(os.path.join(output_dir, f'{cell_id}_cycles.csv'), index=False)
        logger.info(f"Saved processed dataset from cleaned_dataset to {output_dir}")
        
    return master_df
