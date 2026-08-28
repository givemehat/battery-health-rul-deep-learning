# Week 2 — Day 2: Practical Data Exploration & Baseline Modeling Experiment

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Project  
**Track**: Group O4 — Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)  
**Author**: Rajnish Singh  
**Submission Date**: Week 2, Day 2  

---

## 1. Practical Dataset Exploration & Quality Audit

### 1.1 Dataset Scope & Verification
We downloaded and parsed the official **NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset** (ARC-FY08Q4 and ARC 25–56), extracting tabular cycle-level telemetry across **34 `.mat` files** comprising **2,744 total discharge cycles**.

| Dataset Metric | Verified Count / Value | Notes |
| :--- | :---: | :--- |
| **Total Raw `.mat` Files** | 34 files | All 34 NASA battery aging cells parsed |
| **Primary Research Benchmark Cells** | 4 cells (`B0005`, `B0006`, `B0007`, `B0018`) | Exact cells evaluated in *Qiu et al. (2024)* |
| **Total Processed Discharge Cycles** | 2,744 cycles | Charge & impedance cycles isolated |
| **Missing / Null Feature Values** | 0 (0.00%) | Handled via forward-fill & backward-fill |
| **Data Leakage Risk** | Strictly Eliminated | Scalers fit only on training cells (`B0005`, `B0006`) |

### 1.2 Data Quality Audit Table

| Cell ID | Cycle Count | Initial Capacity ($C_0$) | EOL Threshold | EOL Cycle ($k_{\text{EOL}}$) | Ambient Temp | Status / Audit Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `B0005` | 168 | 1.856 Ah | 1.40 Ah | Cycle 124 | 24°C | Clean regular CC-CV discharge profile |
| `B0006` | 168 | 2.035 Ah | 1.40 Ah | Cycle 109 | 24°C | Accelerated degradation rate |
| `B0007` | 168 | 1.891 Ah | 1.50 Ah | Cycle 148 | 24°C | Slow fade; capacity remained $>1.40\text{ Ah}$ |
| `B0018` | 132 | 1.855 Ah | 1.40 Ah | Cycle 96 | 24°C | Rapid fade with marked regeneration spikes |
| `AUV_SIM`| 200 | 2.020 Ah | 1.40 Ah | Cycle 142 | 4°C–20°C | Simulated marine thermal & thruster load profile |

---

## 2. Leakage-Safe Cell-Wise Temporal Split

To guarantee zero data leakage between train and test sets, we implemented a **Cell-Wise Zero-Shot Evaluation Split**:

```
Total Processed Dataset (NASA Benchmark)
 ├── Training Set (Cells B0005 & B0006):    308 Sequence Windows (L=15)
 ├── Validation Set (Cell B0007):          154 Sequence Windows (L=15)
 └── Out-of-Sample Test Set (Cell B0018):   118 Sequence Windows (L=15)
```

- **Feature Scaler**: `MinMaxScaler(feature_range=(0, 1))` fitted **strictly** on the training cells (`B0005`, `B0006`) and applied out-of-sample to `B0007` and `B0018`.
- **Sequence Construction**: Temporal sliding window of length $L = 15$ cycles. The target $\text{RUL}_k$ corresponds to the cycle at the end of the sliding window.

---

## 3. Baseline Experiments & Comparative Results

We implemented, trained, and benchmarked 8 distinct prognostic models on the zero-shot unseen test battery cell **NASA B0018**:

```
================================================================================
MODEL BENCHMARK COMPARISON TABLE (EVALUATION ON TEST CELL B0018)
================================================================================
| Model Architecture                                      |    MAE |   RMSE |   MAPE (%) |    R2 |   Max Error |   Params |   Train Time (s) |   Inference (ms/sample) |
|---------------------------------------------------------|--------|--------|------------|-------|-------------|----------|------------------|-------------------------|
| Temporal Convolutional Network (TCN)                    |  5.050 |  6.049 |     41.27% | 0.952 |      13.084 |  107,554 |            5.200 |                   1.541 |
| Random Forest Regressor                                 |  7.100 |  7.947 |     43.62% | 0.916 |      18.210 |   50,000 |            0.360 |                   0.050 |
| Proposed Hybrid SOTA (CEEMDAN-TCN-BiLSTM-DualAttention) | 14.397 | 15.809 |    109.87% | 0.669 |      22.050 |  293,090 |            4.960 |                   0.277 |
| Temporal Transformer                                    | 14.507 | 16.041 |    115.68% | 0.659 |      23.317 |   67,714 |           11.390 |                   1.440 |
| Empirical Double-Exp Baseline                           | 17.432 | 19.196 |    133.88% | 0.512 |      22.000 |        4 |            0.010 |                   0.104 |
| BiLSTM-Attention                                        | 17.536 | 20.209 |    142.11% | 0.459 |      30.944 |  220,674 |            9.410 |                   1.349 |
| Standard LSTM                                           | 20.637 | 22.721 |    158.05% | 0.316 |      31.405 |   52,610 |           11.300 |                   1.124 |
| Standard GRU                                            | 21.666 | 23.597 |    159.93% | 0.262 |      32.298 |   39,490 |           22.640 |                   0.346 |
================================================================================
```

---

## 4. Observations & Key Insights

1. **Superiority of Temporal Convolutional Networks (TCN)**:
   - TCN achieved the top performance with **MAE = 5.05 cycles** and **$R^2 = 0.952$**.
   - The dilated causal 1D convolutions with exponentially increasing receptive fields allow the network to model localized electrochemical capacity recovery shocks without suffering from the vanishing gradients observed in standard LSTMs.
2. **Empirical Baseline Failure Under Capacity Regeneration**:
   - The classical double-exponential empirical baseline ($C(k) = a e^{bk} + c e^{dk}$) assumes monotonic capacity decay. When capacity spikes occur after rest intervals, the empirical curve overestimates degradation, yielding a large MAE of $17.43\text{ cycles}$.
3. **Thermal Rise Signature ($\Delta T$)**:
   - Cell peak temperature rise ($\Delta T = T_{\text{max}} - T_{\text{start}}$) showed a strong positive correlation ($r = 0.88$) with cycle index, serving as a leading physical indicator of internal resistance growth.
4. **Degradation-Stage Robustness**:
   - In Early-Stage ($1\text{--}60\text{ cycles}$), TCN achieves an outstanding **MAE of 2.73 cycles** ($R^2 = 0.886$).
   - In Mid-Stage ($61\text{--}120\text{ cycles}$), capacity regeneration increases error to $\text{MAE} = 6.86\text{ cycles}$.
   - In Late-Stage ($>120\text{ cycles}$), near critical EOL, maximum error remains tightly bounded at $7.02\text{ cycles}$.

---

## 5. Challenges Encountered & Solutions

| Challenge | Impact | Technical Solution |
| :--- | :--- | :--- |
| **NumPy 2.x `trapz` Removal** | `np.trapz` deprecated in NumPy 2.0+ causing integration runtime crash | Replaced with `scipy.integrate.trapezoid` and robust custom numerical integration fallback |
| **Non-Monotonic Capacity Noise** | Capacity recovery after rest intervals distorts standard loss convergence | Implemented CEEMDAN / Savitzky-Golay signal decomposition to decouple trend from regeneration shocks |
| **Cross-Cell Initial Capacity Offset** | $C_0$ variations between cells ($1.85\text{ Ah}$ vs $2.03\text{ Ah}$) | Built multi-feature sliding window representation combining normalized SOH, $\Delta C_k$, and voltage std |
| **Computational Footprint for AUVs** | Large models exceed onboard microcontrollers | Optimized TCN architecture to $107\text{k parameters}$ with $<1.6\text{ ms}$ inference latency on Apple MPS/CPU |

---

## 6. Next Steps for Mentor Review (Next 3-Day Plan)

1. **Hyperparameter Optimization & CEEMDAN Mode Tuning**: Fine-tune the sub-network weightings in the Hybrid CEEMDAN-TCN-BiLSTM architecture to surpass standalone TCN across all operational stages.
2. **Multi-Step Horizon Forecasting**: Extend predictions from single-step RUL to multi-step recursive trajectory rollout ($k \to k+30$).
3. **Physics-Informed Loss Regularization**: Integrate electrochemical thermodynamic bounds into the loss function to enforce strictly negative long-term capacity slope constraints.
