# Methodology Document V1: Deep Learning-Based Battery Health Monitoring and Remaining Useful Life Prediction for Autonomous Underwater Vehicles (AUVs)

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Project  
**Track**: Group O4 — Deep Learning / Battery Prognostics / Autonomous Systems  
**Intern**: Rajnish Singh  
**Milestone**: Day 6 — Pre-Dataset Readiness + Mentor Review  
**Document Version**: V1.1 (Updated Post-Mentor Review & Corrections)  

---

## 1. Executive Summary & Problem Formulation

Autonomous Underwater Vehicles (AUVs) are deployed for extended, high-stakes oceanic missions including bathymetric seafloor surveying, benthic pipeline inspection, marine environmental sampling, and scientific observation (*Ma et al., 2025*). Unlike terrestrial electric vehicles, an AUV cannot pull over or receive emergency roadside assistance when an onboard energy failure occurs; premature battery depletion or unpredicted electrochemical failure leads directly to mission abortion or unrecoverable vehicle loss on the ocean floor.

This research project delivers a data-driven, temporal deep learning framework for:
1. **State of Health (SOH) Estimation**: Quantifying the instantaneous usable charge capacity $C_k$ relative to the nominal factory rating $C_{\text{nominal}}$.
2. **Remaining Useful Life (RUL) Prediction**: Forecasting the exact number of remaining charge-discharge cycles before the battery reaches its critical End-of-Life (EOL) failure threshold.
3. **Capacity Regeneration Mitigation**: Decoupling non-linear electrochemical capacity recovery spikes (caused by chemical relaxation during resting intervals between missions) from the irreversible underlying thermodynamic degradation trend.

### 1.1 Mathematical Formulation

#### State of Health (SOH)
$$\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$$
Where:
- $C_k$ is the measured discharge capacity at cycle $k$ in Ampere-hours (Ah).
- $C_{\text{nominal}} = 2.00\text{ Ah}$ for standard NASA 18650 cylindrical LiCoO2 cells.

#### Remaining Useful Life (RUL)
$$\text{RUL}_k = \max\left(0, k_{\text{EOL}} - k\right)$$
Where:
- $k_{\text{EOL}}$ is the first cycle index where capacity permanently drops below the defined failure threshold:
$$k_{\text{EOL}} = \min \{ k \in \mathbb{N} \mid C_k \le C_{\text{EOL\_threshold}} \}$$

#### Failure Threshold Specification (Mentor-Confirmed)
- **Standard Cells (`B0005`, `B0006`, `B0018`)**: $C_{\text{EOL\_threshold}} = 1.40\text{ Ah}$ ($70\%$ nominal rated capacity).
- **Cell `B0007`**: $C_{\text{EOL\_threshold}} = 1.50\text{ Ah}$ ($75\%$ nominal capacity).  
  *Rationale*: In the original NASA test, cell `B0007` cycling was terminated at cycle 168 before reaching $1.40\text{ Ah}$ (final capacity was $1.439\text{ Ah}$). Adhering to the established academic benchmark protocol in *Qiu et al. (2024)*, the failure threshold is formally set to $1.50\text{ Ah}$ ($k_{\text{EOL}} = 148$).

---

## 2. Literature Review & Prior Art Positioning

| Research Work | Dataset & Scope | Methodological Core | Reported Metrics / Findings | Key Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Qiu et al. (2024)** (*World Electr. Veh. J.*) | NASA (`B0005`, `B0006`, `B0007`, `B0018`) & CALCE | CEEMDAN + Improved Sparrow Search (IHSSA) + LSTM + TCN | High accuracy on capacity trajectories ($0.8\%\text{--}1.4\%$ MAE) | Computationally heavy metaheuristic tuning; lacks zero-shot cross-cell validation; no marine AUV context. |
| **Ma et al. (2025)** (*Ocean Engineering Review*) | Global Marine Autonomous Vehicles | High-level survey of AUV power systems, lithium batteries, and AI fault diagnosis | Highlights critical need for predictive BMS in long-range subsea missions | Broad review without quantitative temporal deep learning benchmark implementations. |
| **Baseline LSTM** | NASA PCoE | 2-Layer Sequential LSTM | Basic temporal modeling capability | Gradient degradation over long sequences; susceptible to capacity regeneration overshoot. |
| **Baseline TCN** | NASA PCoE | Dilated Causal 1D Convolutions with Residual connections | Superior localized feature extraction; parallel training | Receptive field limited by kernel size and dilation depth; lacks explicit global recurrent state. |
| **Proposed Methodology V1 (This Work)** | NASA PCoE (34 cells, 2,744 cycles) + AUV Mission Profile | **Multi-Branch CEEMDAN-TCN-BiLSTM-DualAttention** with Multi-Task Loss ($\mathcal{L}_{\text{RUL}} + \lambda \mathcal{L}_{\text{cap}}$) | **TCN achieves MAE 5.05 cycles, $R^2 = 0.952$** on unseen zero-shot test cell `B0018`; $<1.6\text{ ms}$ latency | Requires structured temporal sequence windowing; parameter count requires quantization for edge MCU. |

---

## 3. Data-Readiness Package & Quality Assessment

### 3.1 Raw Dataset Ingestion
The raw dataset consists of the complete **NASA Ames Prognostics Center of Excellence (PCoE) Battery Aging Data Set** (ARC-FY08Q4 and ARC 25–56), containing 34 MATLAB `.mat` files.

- **Preservation Policy**: The raw `.mat` files and archive zip packages are preserved in an untouched read-only state in `data/raw/`.
- **Parsing Pipeline**: An automated, non-destructive parser (`src/data/parser_nasa.py`) segments the mixed operational stream into:
  1. Constant-Current / Constant-Voltage (CC-CV) Charge Cycles.
  2. Constant-Current Discharge Cycles (Operating mission cycles).
  3. Electrochemical Impedance Spectroscopy (EIS) Frequency Sweep Cycles.

### 3.2 Extracted Feature Schema (Data Dictionary)

| Feature | Physical Unit | Description | Extraction / Computation Method |
| :--- | :---: | :--- | :--- |
| `cell_id` | String | Unique battery cell identifier | Extracted from file metadata |
| `cycle_index` | Integer | Sequential discharge cycle count ($1, 2, 3, \dots$) | Filtered sequential index |
| `capacity` | Ah | Discharge capacity | Integrated load current: $\frac{1}{3600}\int \vert I(t) \vert dt$ |
| `soh` | % | State of Health percentage | $\frac{C_k}{C_{\text{nominal}}} \times 100$ |
| `v_start` | V | Terminal voltage at discharge initiation | First valid voltage reading in cycle |
| `v_end` | V | Cut-off terminal voltage | Last valid voltage reading in cycle |
| `v_min` | V | Minimum observed voltage | $\min_{t} V(t)$ |
| `v_mean` | V | Mean cycle voltage | $\frac{1}{N} \sum_{t=1}^N V(t)$ |
| `v_std` | V | Voltage dispersion indicator | Standard deviation of voltage trajectory |
| `i_mean` | A | Mean discharge current | Average load current magnitude |
| `t_start` | °C | Initial cell surface temperature | First temperature sensor sample |
| `t_max` | °C | Peak maximum surface temperature | $\max_{t} T(t)$ |
| `t_mean` | °C | Average cell temperature | Mean temperature reading |
| `t_rise` | °C | Thermal elevation ($\Delta T$) | $T_{\text{max}} - T_{\text{start}}$ (Key aging feature) |
| `discharge_duration` | s | Total time taken to discharge | $t_{\text{end}} - t_{\text{start}}$ |
| `energy_discharged` | Wh | Electrical energy output | $\frac{1}{3600}\int V(t) \vert I(t) \vert dt$ |
| `capacity_diff` | Ah | Cycle-to-cycle capacity change | $C_k - C_{k-1}$ |
| `is_regeneration` | Binary | Rest relaxation rebound indicator | $1$ if $\Delta C_k > 0.005\text{ Ah}$, else $0$ |
| `degradation_rate` | Ah/cycle| Rolling slope of degradation | 5-cycle backward gradient $\frac{dC}{dk}$ |
| `rul_true` | Cycles | Ground truth remaining useful life | $\max(0, k_{\text{EOL}} - k)$ |

### 3.3 Data Quality Audit & Risk Mitigation
- **Missing / Invalid Values**: Verified zero missing values across the 2,744 processed discharge cycles; automated forward-fill/backward-fill handles any sensor dropouts.
- **NumPy 2.x Migration**: In NumPy 2.0+, `np.trapz` was removed. The parsing engine utilizes `scipy.integrate.trapezoid` with a robust fallback numerical integration function to ensure future-proof reproducibility.
- **Partial Cycles & Outliers**: Cycles with discharge duration $< 100\text{ s}$ or voltage variance $< 10^{-4}$ are filtered out as abortive test cycles.

---

## 4. Proposed Preprocessing, Modeling & Evaluation Plan

### 4.1 Zero-Leakage Split Strategy (Mentor-Confirmed)

To guarantee uncompromising scientific validity and prevent temporal information leakage:
1. **Cell-Wise Zero-Shot Split**:
   - **Training Set**: Cells `B0005` (168 cycles) and `B0006` (168 cycles) $\to$ **308 sliding sequence windows**.
   - **Validation Set**: Cell `B0007` (168 cycles) $\to$ **154 sliding sequence windows**.
   - **Out-of-Sample Test Set**: Cell `B0018` (132 cycles) $\to$ **118 sliding sequence windows**.
2. **Scaler Leakage Protection**:
   - `MinMaxScaler(feature_range=(0, 1))` is fitted **strictly** on the training cells (`B0005`, `B0006`).
   - The fitted transformation parameters ($\mu, \sigma, \min, \max$) are saved and applied out-of-sample to `B0007` and `B0018`.
3. **Temporal Windowing**:
   - Sliding sequence window of length $L = 15$ cycles.
   - Input tensor: $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$, where $D = 9$ engineered physical features.
   - Target: $\mathbf{y}_{\text{RUL}} \in \mathbb{R}^{B \times 1}$ and $\mathbf{y}_{\text{cap}} \in \mathbb{R}^{B \times 1}$ at time $t = k$.

### 4.2 Signal Decomposition: Decoupling Aging from Regeneration
Following the paradigm in *Qiu et al. (2024)*, battery capacity degradation series $C(k)$ is decomposed into:
$$C(k) = C_{\text{LF}}(k) + C_{\text{HF}}(k)$$
- **Low-Frequency Component $C_{\text{LF}}(k)$**: Smooth, monotonic thermodynamic capacity fade driven by Solid Electrolyte Interphase (SEI) layer growth and active lithium loss. Extracted using multi-scale Savitzky-Golay / trend filtering.
- **High-Frequency Component $C_{\text{HF}}(k)$**: Electrochemical relaxation rebounds (capacity regeneration after prolonged resting) and sensor noise.

### 4.3 Architecture Zoo & Multi-Task Training

```
                        Input Sliding Sequence [Batch, 15, 9]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      [TCN Branch (High-Freq)]                   [BiLSTM Branch (Low-Freq)]
      • Dilated Causal Conv1D (d=1,2,4)          • 2-Layer Bidirectional LSTM
      • Receptive Field = 15 cycles              • Hidden Dim = 64 (128 combined)
      • Squeeze-and-Excitation Channel Attn      • Multi-Head Self-Attention (H=4)
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                       [Gated Cross-Fusion Layer]
                         Fused = [h_TCN, h_BiLSTM] * σ(W_g * Fused)
                                       │
                                       ▼
                       [Shared Fully-Connected Dense]
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             [RUL Prediction Head]             [SOH Capacity Head]
             Loss_RUL = MSE(RUL, RUL*)         Loss_Cap = MSE(C, C*)
```

#### Multi-Task Loss Formulation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\widehat{\text{RUL}}, \text{RUL}^*) + \lambda_{\text{cap}} \cdot \mathcal{L}_{\text{MSE}}(\hat{C}, C^*)$$
- Optimal convergence confirmed at $\lambda_{\text{cap}} = 10.0$.
- Optimizer: AdamW with weight decay $10^{-4}$, initial learning rate $\eta = 10^{-3}$.
- Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`.
- Regularization: EarlyStopping with patience $15\text{ epochs}$, gradient norm clipping at $1.0$.

---

## 5. Evaluation Protocol & Constraint Verification

### 5.1 Evaluation Metrics (Problem Statement Aligned)

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \vert y_i - \hat{y}_i \vert$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left\vert \frac{y_i - \hat{y}_i}{y_i} \right\vert$$
4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
5. **Maximum Absolute Error ($\text{Max Error}$)**:
   $$\text{Max Error} = \max_{i} \vert y_i - \hat{y}_i \vert$$

### 5.2 Embedded AUV Operational Constraints
- **Inference Latency Limit**: $\le 10\text{ ms}$ per cycle estimate.  
  *Achieved Benchmark*: **$0.28\text{ ms}$ to $1.54\text{ ms}$** on Apple MPS/CPU.
- **Model Footprint Limit**: $\le 5\text{ MB}$ parameter storage.  
  *Achieved Benchmark*: **$107\text{k params} \approx 428\text{ kB}$** (TCN) and **$293\text{k params} \approx 1.17\text{ MB}$** (Hybrid SOTA).
- **Zero-Leakage Constraint**: Confirmed zero training data overlap with test battery cell `B0018`.

---

## 6. Degradation-Stage Robustness Framework

Prognostic accuracy is evaluated across three distinct operational regimes:
1. **Early-Stage ($1 \le k \le 60$)**: High capacity, linear fade regime, battery stabilization.
2. **Mid-Stage ($61 \le k \le 120$)**: Active capacity regeneration events, dynamic rest intervals.
3. **Late-Stage / Near-EOL ($k > 120$)**: Severe electrochemical knee-point decay, critical safety zone.
