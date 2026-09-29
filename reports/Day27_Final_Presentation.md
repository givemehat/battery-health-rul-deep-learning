# 6-Slide Executive Research Presentation Pack
### TIH IIT Guwahati — 4-Week Online Internship Research Programme (Group O4)
**Project Title**: Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)  
**Intern**: Rajnish Singh  

---

### 🖥️ Slide 1: Research Motivation & Problem Formulation
- **Domain Context**: Autonomous Underwater Vehicles (AUVs) operate in deep-sea environments where battery exhaustion or unexpected degradation leads to catastrophic vehicle loss (*Ma et al., 2025*).
- **Core Objectives**:
  1. Accurate real-time State of Health (SOH) estimation: $\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$.
  2. Non-linear Remaining Useful Life (RUL) trajectory forecasting: $\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$.
- **Key Challenges**: Non-monotonic capacity regeneration (electrochemical relaxation after rest intervals), bathymetric thermal swings ($4^\circ\text{C}$ to $20^\circ\text{C}$), and strict onboard edge compute limits ($< 5.0\text{ ms}$ latency).

---

### 🖥️ Slide 2: Data-Readiness Architecture & Leakage-Safe Pipeline
- **Dataset Scale**: Ingested and structured all 34 NASA PCoE battery aging cells from `cleaned_dataset` (**7,565 total operations**, **2,794 discharge cycles**).
- **Physical & Electrochemical Features**: Terminal voltage drop ($V_{\text{mean}}, V_{\text{std}}$), thermal elevation ($\Delta T = T_{\text{max}} - T_{\text{start}}$), energy delivered ($\int V \cdot I dt$), and chemical relaxation indicators ($\Delta C_k > 0.005\text{ Ah}$).
- **Strict Zero-Leakage Split**:
  - **Training Set**: Cells `B0005` & `B0006` (fit scalers strictly here).
  - **Validation Set**: Cell `B0007` (hyperparameter tuning).
  - **Unseen Test Set**: Cell `B0018` (zero-shot generalization test).

---

### 🖥️ Slide 3: Multi-Task Deep Learning & Signal Decomposition
- **Signal Decoupling**: CEEMDAN / Savitzky-Golay decomposition separating low-frequency thermodynamic capacity fade ($C_{\text{LF}}$) from high-frequency regeneration shocks ($C_{\text{HF}}$).
- **Architecture Zoo (8 Benchmarked Models)**:
  - Baselines: Empirical Double-Exponential ($C(k) = ae^{bk} + ce^{dk}$), Random Forest Regressor.
  - Deep Recurrent & Convolutions: Standard LSTM, GRU, Dilated Causal TCN, BiLSTM-Attention, Temporal Transformer, and Proposed Hybrid SOTA (CEEMDAN-TCN-BiLSTM-DualAttn).
- **Multi-Task Loss Regularization**:
  $$\mathcal{L}_{\text{total}} = \text{MSE}(\widehat{\text{RUL}}, \text{RUL}^*) + 10.0 \cdot \text{MSE}(\hat{C}, C^*)$$
  Constrains latent representations to physical capacity dynamics rather than trivial integer countdowns.

---

### 🖥️ Slide 4: Experimental Benchmarks & Quantitative Results
Evaluated on completely unseen test battery cell **NASA B0018**:

| Model Architecture | MAE (Cycles) | RMSE (Cycles) | MAPE (%) | $R^2$ Score | Max Error | Inference Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Convolutional Network (TCN)** | **5.050** | **6.049** | **41.27%** | **0.952** | **13.08** | **1.54 ms** | **428 kB** |
| **Random Forest Regressor** | **7.100** | **7.947** | **43.62%** | **0.916** | **18.21** | 0.05 ms | 1.8 MB |
| **Proposed Hybrid SOTA** | 14.397 | 15.809 | 109.87% | 0.669 | 22.05 | 0.28 ms | 1.1 MB |
| **Temporal Transformer** | 14.507 | 16.041 | 115.68% | 0.659 | 23.32 | 1.44 ms | 270 kB |
| **Empirical Baseline** | 17.432 | 19.196 | 133.88% | 0.512 | 22.00 | 0.10 ms | < 1 kB |
| **BiLSTM-Attention** | 17.536 | 20.209 | 142.11% | 0.459 | 30.94 | 1.35 ms | 880 kB |
| **Standard LSTM** | 20.637 | 22.721 | 158.05% | 0.316 | 31.41 | 1.12 ms | 210 kB |

*Key Finding: TCN dilated causal 1D convolutions capture both local regeneration shocks and long-range degradation trends with 95.2% variance explained.*

---

### 🖥️ Slide 5: Robustness Analysis & Degradation Stage Breakdown
- **Operational Stage Robustness (TCN Model)**:
  - **Early-Stage (Cycles 1–60)**: **MAE = 2.73 cycles**, **RMSE = 4.49 cycles**, **$R^2 = 0.886$**.
  - **Mid-Stage (Cycles 61–120 / Regeneration Dynamic)**: **MAE = 6.86 cycles**, **RMSE = 7.18 cycles**, **$R^2 = 0.649$**.
  - **Late-Stage / Near-EOL Critical (Cycles > 120)**: **MAE = 4.90 cycles**, **Max Error = 7.02 cycles**.
- **Error Envelope**: Over **85% of predictions fall strictly within the high-confidence $\pm 5$ cycle band**, enabling early mission abort decisions before irreversible cell failure.

---

### 🖥️ Slide 6: Edge Hardware Deployment & Research Roadmap
- **Embedded Compute Feasibility**: Verified inference latency of **$1.54\text{ ms}$** and memory footprint of **$428\text{ kB}$**, comfortably satisfying onboard microcontrollers (STM32 / Raspberry Pi CM4 / NVIDIA Jetson Orin Nano).
- **Completed Milestones**:
  1. Data Quality & Zero-Leakage Pipeline (Day 08 – Day 14).
  2. Model Zoo Development & Benchmarking (Day 15 – Day 21).
  3. Frozen Inference Checkpoints & Diagnostics (Day 22 – Day 26).
  4. Complete 366-Cell Master Pipeline & 3-Tier Notebook Hierarchy.
- **Next Steps**: Hardware-in-the-loop (HIL) marine tank testing and multi-cell pack thermal balancing.
