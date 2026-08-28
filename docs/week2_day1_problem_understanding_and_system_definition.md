# Week 2 — Day 1: Problem Understanding & System Definition

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Project  
**Track**: Group O4 — Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)  
**Author**: Rajnish Singh  
**Submission Date**: Week 2, Day 1  

---

## 1. Project Objective & Technical Problem Statement

### 1.1 Problem Statement (Rewritten in Technical Form)
Autonomous Underwater Vehicles (AUVs) operate in remote, hostile, and thermally variable marine environments where energy autonomy is paramount (*Ma et al., 2025*). Lithium-ion battery packs serve as the sole onboard power source for vehicle propulsion (thrusters), acoustic payloads, navigation sensors, and communications. Unexpected capacity depletion or accelerated electrochemical degradation poses severe operational risks, potentially resulting in mission abortion, unrecoverable subsea stranding, or catastrophic vehicle loss.

The technical objective of this project is to construct a robust, data-driven, temporal Deep Learning framework capable of:
1. **Accurately Estimating State of Health (SOH)**: Quantifying the degradation level and loss of usable charge capacity relative to the nominal factory rating.
2. **Forecasting Remaining Useful Life (RUL)**: Predicting the exact number of remaining operational charge-discharge cycles before the cell's capacity crosses the critical End-of-Life (EOL) failure threshold ($70\%$ rated capacity or $1.40\text{ Ah}$ for standard NASA 18650 cells; $1.50\text{ Ah}$ for B0007 per *Qiu et al., 2024*).
3. **Handling Non-Monotonic Capacity Regeneration**: Modeling the electrochemical relaxation rebound phenomenon occurring during resting intervals between operational missions without cumulative drift.

---

## 2. Mathematical System Formulation

### 2.1 State of Health (SOH)
The State of Health at discharge cycle $k$ is defined as the ratio of current usable discharge capacity $C_k$ to the initial nominal rated capacity $C_{\text{nominal}}$:

$$\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$$

Where for NASA 18650 LiCoO2 cells, $C_{\text{nominal}} = 2.00\text{ Ah}$.

### 2.2 Remaining Useful Life (RUL)
The true Remaining Useful Life at cycle $k$ represents the cycle distance to the End-of-Life cycle $k_{\text{EOL}}$:

$$\text{RUL}_k = \max\left(0, k_{\text{EOL}} - k\right)$$

Where $k_{\text{EOL}}$ is the first cycle index satisfying:

$$k_{\text{EOL}} = \min \{ k \mid C_k \le C_{\text{EOL\_threshold}} \}$$

- For cells `B0005`, `B0006`, `B0018`: $C_{\text{EOL\_threshold}} = 1.40\text{ Ah}$ ($70\%$ nominal capacity).
- For cell `B0007`: $C_{\text{EOL\_threshold}} = 1.50\text{ Ah}$ ($75\%$ nominal capacity, due to test termination before $1.40\text{ Ah}$).

---

## 3. Input / Output Specifications

| Component | Variable Name | Unit | Source / Physical Meaning |
| :--- | :--- | :---: | :--- |
| **Input Feature 1** | $C_k$ (Discharge Capacity) | Ah | Coulomb counting / current integration $\int I(t) dt$ |
| **Input Feature 2** | $\text{SOH}_k$ | % | Normalized capacity percentage |
| **Input Feature 3** | $\bar{V}_k$ (Mean Voltage) | V | Average terminal voltage across discharge |
| **Input Feature 4** | $\sigma_{V, k}$ (Voltage Std) | V | Dispersion of voltage drop curve |
| **Input Feature 5** | $\bar{T}_k$ (Mean Temperature)| °C | Average cell surface thermal reading |
| **Input Feature 6** | $\Delta T_k$ (Thermal Rise) | °C | Thermal surge $\Delta T = T_{\text{max}} - T_{\text{start}}$ |
| **Input Feature 7** | $t_{\text{dis}, k}$ (Duration) | s | Total time taken to discharge to cut-off |
| **Input Feature 8** | $E_{\text{dis}, k}$ (Energy) | Wh | Electrical energy output $\int V(t) \cdot I(t) dt$ |
| **Input Feature 9** | $\Delta C_k$ (Capacity Delta)| Ah | Difference $C_k - C_{k-1}$ (regeneration tag) |
| **Target Output 1** | $\widehat{\text{RUL}}_k$ | cycles | Predicted remaining charge-discharge cycles |
| **Target Output 2** | $\hat{C}_{k+1}$ (Multi-task SOH)| Ah | Forecasted next-cycle capacity |

---

## 4. End-to-End Software & Modeling Pipeline

```
  [Raw Telemetry (.mat / .csv)]
               │
               ▼
  [Data Ingestion & Cycle Segmentation] (Separate CC-CV Charge, Discharge & Impedance)
               │
               ▼
  [Feature Extraction & Quality Audit] (Extract V, I, T stats, Energy, Duration, SOH)
               │
               ▼
  [Signal Decomposition] (CEEMDAN / Trend Filter: Trend vs Regeneration Rebound)
               │
               ▼
  [Leakage-Safe Temporal Windowing] (Sliding Sequence L=15, Scaler fit strictly on Train)
               │
               ▼
  [Prognostic Model Zoo]
  ┌─────────────────────────────────────────────────────────────┐
  │  • Empirical Exponential Baseline                           │
  │  • Random Forest Regressor                                  │
  │  • Standard LSTM & GRU Networks                             │
  │  • Temporal Convolutional Network (TCN)                     │
  │  • BiLSTM + Multi-Head Self-Attention                       │
  │  • Temporal Transformer (Patch Positional Encoder)          │
  │  • Proposed SOTA Hybrid (CEEMDAN-TCN-BiLSTM-DualAttn)       │
  └─────────────────────────────────────────────────────────────┘
               │
               ▼
  [Multi-Task Training Engine] (L = L_RUL + λ * L_Cap, EarlyStopping, ReduceLROnPlateau)
               │
               ▼
  [Evaluation & Robustness Suite] (Cross-Cell Zero-Shot, Degradation Stages, Latency Benchmark)
               │
               ▼
  [AUV Battery Management System (BMS) Decision Engine]
```

---

## 5. Technology Stack & Toolchain

- **Core Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch 2.x (with MPS Apple Silicon & CUDA acceleration)
- **Scientific Computing & Signal Processing**: NumPy 2.x, SciPy 1.18+ (trapezoidal integration, Savitzky-Golay filtering)
- **Data Engineering**: Pandas 3.x
- **Visualization**: Matplotlib 3.11+, Seaborn 0.13+
- **Machine Learning & Preprocessing**: Scikit-Learn 1.9+
- **Version Control & Documentation**: Git, GitHub CLI (`gh`), Jupyter Notebooks, Markdown

---

## 6. Technical Questions & Assumptions for Mentor Review

1. **Failure Threshold Consistency**: *In NASA dataset cell B0007, the test was discontinued before reaching 1.40 Ah. We set the EOL threshold for B0007 to 1.50 Ah as suggested by Qiu et al. (2024). Should this standard be maintained across all cross-validation folds?*
2. **Cross-Cell Domain Shift**: *Different battery cells exhibit varying manufacturing tolerances and initial capacities (e.g. B0005 initial capacity = 1.85 Ah vs B0006 = 2.03 Ah). How can we best normalize feature representations to prevent cell-bias transfer?*
3. **Capacity Regeneration Weighting**: *Rest periods between cycles introduce sudden positive capacity jumps ($\Delta C > 0$). Should the loss function penalize overestimation during regeneration differently than underestimation near critical EOL?*
4. **Sampling Rate vs Embedded Latency**: *For practical AUV BMS deployment, what is the maximum acceptable inference latency per cycle estimation (our benchmark achieves $<1.5\text{ ms}$)?*
5. **Multi-Task Loss Balancing ($\lambda$)**: *We observed optimal convergence when $\lambda_{\text{capacity}} = 10.0$ relative to $\mathcal{L}_{\text{RUL}}$. What are the mentor's recommendations on dynamic loss weighting (e.g., GradNorm or Uncertainty Weighting)?*
6. **Partial Discharge Scenarios**: *NASA telemetry operates on fixed cut-off voltages (2.7V, 2.5V, 2.2V). In real AUV missions, batteries experience shallow/partial cycling. How can we augment the model for variable depth-of-discharge (DoD)?*
7. **Thermal Extreme Sensitivity**: *In deep-sea operations ($4^\circ\text{C}$ bathymetric ocean floor), internal resistance increases. How should ambient temperature features be integrated into the temporal attention layers?*
8. **Evaluation Metric Priority**: *While RMSE penalizes large outlier errors, MAE gives a linear cycle error. In marine mission planning, is maximum worst-case error ($\text{Max Error}$) a more critical safety metric?*

---

## 7. Success Criteria & Expected Deliverables

- **Quantitative Target**: RUL Mean Absolute Error (MAE) $< 6.0\text{ cycles}$ on unseen test battery cell (`B0018`) in zero-shot cross-cell evaluation.
- **Robustness Target**: Stable degradation forecasting across Early ($1\text{--}60$), Mid ($61\text{--}120$), and Late ($>120$) operating life stages.
- **Code Quality**: Modular, fully documented, independently executable codebase on GitHub with automated EDA, training, and benchmarking scripts.
