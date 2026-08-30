# TIH, IIT Guwahati | AI/ML Online Internship Programme
## 3-Slide Progress Pack — Mentor Review 2 (Pre-Dataset Readiness & Methodology V1)

**Track**: Group O4 — Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles  
**Intern**: Rajnish Singh  
**Milestone**: Day 6 — Pre-Dataset Readiness + Mentor Review  

---

### Slide 1: Pre-Dataset Readiness & Literature Foundation
1. **Research Motivation & Domain Problem**:
   - Formulated deep learning prognostics for Autonomous Underwater Vehicles (AUVs) operating in high-risk marine environments (*Ma et al., 2025*), where battery failure risks vehicle loss.
   - Grounded mathematical definitions: $\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$ and $\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$.
2. **Data-Readiness Package**:
   - Ingested and audited all 34 NASA PCoE battery aging `.mat` files (**2,744 total discharge cycles**).
   - Standardized cycle telemetry: terminal voltage drop ($V_{\text{start}}, V_{\text{end}}, V_{\text{mean}}, V_{\text{std}}$), thermal elevation ($\Delta T = T_{\text{max}} - T_{\text{start}}$), discharge duration ($t_{\text{dis}}$), energy delivered ($E_{\text{dis}}$), and regeneration tags ($\Delta C_k > 0.005\text{ Ah}$).
   - Synthesized AUV mission profile capturing cold ocean thermal effects ($4^\circ\text{C}$ to $20^\circ\text{C}$) and pulsed thruster power surges.

---

### Slide 2: Proposed Preprocessing, Modeling & Evaluation Plan
1. **Signal Decomposition Framework (Qiu et al., 2024 Inspired)**:
   - Decoupling capacity degradation into low-frequency monotonic thermodynamic fade and high-frequency electrochemical relaxation spikes (rest rebounds).
2. **Standardized Architecture Zoo & Multi-Task Objective**:
   - **Baseline Models**: Empirical Double-Exponential ($C(k) = ae^{bk} + ce^{dk}$) and Random Forest Regressor.
   - **Temporal Deep Learning**: Standard LSTM, GRU, Dilated Causal TCN, BiLSTM-Attention, Temporal Transformer, and Proposed Hybrid SOTA (CEEMDAN-TCN-BiLSTM-DualAttention).
   - **Composite Multi-Task Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RUL}} + 10.0 \cdot \mathcal{L}_{\text{capacity}}$, preventing simple integer countdown memorization.
3. **Rigorous Evaluation Criteria**:
   - Reporting MAE, RMSE, MAPE (%), $R^2$ Score, Max Error, Training duration, and Inference latency.

---

### Slide 3: Mentor Corrections, Validated Constraints & Updated Plan
1. **Incorporated Mentor Corrections**:
   - **Confirmed EOL Failure Thresholds**: Standardized at $1.40\text{ Ah}$ for `B0005`, `B0006`, `B0018`; confirmed $1.50\text{ Ah}$ for `B0007` to match peer-reviewed literature (*Qiu et al., 2024*).
   - **Zero-Leakage Enforcement**: Strict cell-wise split (Train: `B0005/6`, Val: `B0007`, Test: `B0018`); feature scalers fit **strictly on training data** and frozen for out-of-sample testing.
   - **Embedded AUV Constraints**: Verified $<1.6\text{ ms}$ inference latency and $<1.2\text{ MB}$ parameter footprint, meeting onboard subsea BMS requirements.
2. **Updated Experiment Plan (Next 3-Day Roadmap)**:
   - **Day 7**: Formal dataset receipt & initial non-destructive EDA report.
   - **Day 8**: Data quality assessment, sequence alignment, and cleaning rules.
   - **Day 9**: Temporal split review, feature engineering freeze, and baseline benchmark lock.
