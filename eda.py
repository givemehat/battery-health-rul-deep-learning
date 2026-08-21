import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.parser_nasa import load_all_nasa_cells, parse_nasa_mat_file, EOL_THRESHOLDS
from src.data.auv_simulator import generate_auv_mission_profile
from src.features.decomposition import decompose_capacity_series
from src.utils.helpers import setup_logger

logger = setup_logger('BatteryEDA')

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

def run_exploratory_data_analysis(data_dir: str = 'data'):
    logger.info('Starting Comprehensive Exploratory Data Analysis (EDA)...')
    
    # 1. Load or parse NASA cells
    processed_nasa_path = os.path.join(data_dir, 'processed', 'nasa_battery_cycles.csv')
    if os.path.exists(processed_nasa_path):
        df_nasa = pd.read_csv(processed_nasa_path)
    else:
        df_nasa = load_all_nasa_cells(os.path.join(data_dir, 'raw'), output_dir=os.path.join(data_dir, 'processed'))
        
    df_auv = generate_auv_mission_profile(num_cycles=200)
    
    logger.info(f'NASA dataset: {len(df_nasa)} total cycle records across {df_nasa["cell_id"].nunique() if len(df_nasa) > 0 else 0} cells.')
    
    # ----------------------------------------------------
    # FIGURE 1: Capacity Degradation Curves & EOL Threshold
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    palette = {'B0005': '#1f77b4', 'B0006': '#ff7f0e', 'B0007': '#2ca02c', 'B0018': '#d62728', 'AUV_PACK_CELL_01': '#9467bd'}
    
    if len(df_nasa) > 0:
        for cell_id, group in df_nasa.groupby('cell_id'):
            if cell_id in ['B0005', 'B0006', 'B0007', 'B0018']:
                plt.plot(group['cycle_index'], group['capacity'], label=f'NASA {cell_id} (EOL Thresh: {EOL_THRESHOLDS.get(cell_id, 1.4)} Ah)', color=palette.get(cell_id, None), linewidth=2)
                
    plt.plot(df_auv['cycle_index'], df_auv['capacity'], label='AUV Pack Mission Simulation', color=palette['AUV_PACK_CELL_01'], linewidth=2, linestyle='--')
    
    plt.axhline(1.40, color='red', linestyle=':', label='Standard EOL Failure Threshold (1.40 Ah / 70% SOH)', linewidth=1.8)
    plt.axhline(1.50, color='green', linestyle=':', label='B0007 EOL Failure Threshold (1.50 Ah / 75% SOH)', linewidth=1.8)
    
    plt.title('Figure 1: Battery Discharge Capacity Degradation Curves Across NASA Cells & AUV Mission Profiles', pad=12, fontweight='bold')
    plt.xlabel('Discharge Cycle Index', fontweight='bold')
    plt.ylabel('Discharge Capacity (Ah)', fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_capacity_degradation_curves.png'))
    plt.close()
    logger.info('Saved Fig 1: Capacity Degradation Curves.')
    
    # ----------------------------------------------------
    # FIGURE 2: Capacity Regeneration Phenomenon (Zoom-In)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    if len(df_nasa) > 0 and 'B0005' in df_nasa['cell_id'].values:
        b5 = df_nasa[df_nasa['cell_id'] == 'B0005'].copy().sort_values('cycle_index')
        plt.plot(b5['cycle_index'], b5['capacity'], marker='o', markersize=4, color='#1f77b4', linewidth=1.8, label='B0005 Capacity Trajectory')
        
        # Highlight regeneration spikes
        regen_points = b5[b5['is_regeneration'] == 1]
        plt.scatter(regen_points['cycle_index'], regen_points['capacity'], color='red', s=60, zorder=5, label='Capacity Regeneration Events (Rest Rebound)')
        
        # Zoom in on key region (cycles 40 to 120)
        plt.xlim(30, 130)
        plt.ylim(1.25, 1.85)
        
    plt.title('Figure 2: Non-Monotonic Capacity Regeneration Phenomenon During Resting Intervals (B0005)', pad=12, fontweight='bold')
    plt.xlabel('Cycle Index', fontweight='bold')
    plt.ylabel('Discharge Capacity (Ah)', fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_capacity_regeneration_zoom.png'))
    plt.close()
    logger.info('Saved Fig 2: Capacity Regeneration Zoom.')
    
    # ----------------------------------------------------
    # FIGURE 3: Thermal Elevation & Temperature Rise
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    if len(df_nasa) > 0:
        for cell_id in ['B0005', 'B0006', 'B0007', 'B0018']:
            cell_data = df_nasa[df_nasa['cell_id'] == cell_id]
            if len(cell_data) > 0:
                plt.plot(cell_data['cycle_index'], cell_data['t_max'], label=f'{cell_id} Peak Temperature', linewidth=1.8)
                
    plt.title('Figure 3: Thermal Signature Evolution (Max Cell Temperature Rise with Electrochemical Aging)', pad=12, fontweight='bold')
    plt.xlabel('Discharge Cycle Index', fontweight='bold')
    plt.ylabel('Maximum Cell Temperature (°C)', fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_thermal_evolution.png'))
    plt.close()
    logger.info('Saved Fig 3: Thermal Evolution.')
    
    # ----------------------------------------------------
    # FIGURE 4: Correlation Matrix Heatmap
    # ----------------------------------------------------
    plt.figure(figsize=(9, 7))
    if len(df_nasa) > 0:
        corr_cols = ['capacity', 'soh', 'v_mean', 'v_std', 't_mean', 't_rise', 'discharge_duration', 'energy_discharged', 'rul_true']
        avail_corr = [c for c in corr_cols if c in df_nasa.columns]
        corr_mat = df_nasa[avail_corr].corr()
        sns.heatmap(corr_mat, annot=True, cmap='RdBu_r', fmt='.2f', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Figure 4: Electrochemical & Physical Sensor Correlation Matrix', pad=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig4_feature_correlation_matrix.png'))
        plt.close()
        logger.info('Saved Fig 4: Correlation Matrix.')
        
    # ----------------------------------------------------
    # FIGURE 5: CEEMDAN / Multi-Resolution Decomposition
    # ----------------------------------------------------
    if len(df_nasa) > 0 and 'B0005' in df_nasa['cell_id'].values:
        b5_caps = df_nasa[df_nasa['cell_id'] == 'B0005']['capacity'].values
        low_f, high_f = decompose_capacity_series(b5_caps)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(b5_caps, color='#1f77b4', linewidth=2, label='Raw Capacity Signal y(t)')
        axes[0].set_ylabel('Raw Cap (Ah)', fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].set_title('Figure 5: Signal Decomposition of Battery Degradation Series (Qiu et al. 2024 Framework)', fontweight='bold')
        
        axes[1].plot(low_f, color='#2ca02c', linewidth=2, label='Low-Frequency Trend Component (Thermodynamic Aging)')
        axes[1].set_ylabel('Trend (Ah)', fontweight='bold')
        axes[1].legend(loc='upper right')
        
        axes[2].plot(high_f, color='#d62728', linewidth=1.5, label='High-Frequency Component (Regeneration Peaks + Fluctuation Noise)')
        axes[2].set_ylabel('Residual (Ah)', fontweight='bold')
        axes[2].set_xlabel('Cycle Index', fontweight='bold')
        axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig5_signal_decomposition.png'))
        plt.close()
        logger.info('Saved Fig 5: Signal Decomposition.')
        
    logger.info('EDA Completed successfully! All figures saved in results/figures.')

if __name__ == '__main__':
    run_exploratory_data_analysis()
