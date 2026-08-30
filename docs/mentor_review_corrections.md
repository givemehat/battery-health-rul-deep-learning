# Mentor Review Corrections & Feedback Log (Mentor Review 2)

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Project  
**Track**: Group O4 — Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)  
**Intern**: Rajnish Singh  
**Milestone**: Day 6 — Pre-Dataset Readiness + Mentor Review  
**Review Status**: Complete & Incorporated into Methodology V1  

---

## 1. Executive Review Summary

During **Mentor Review 2**, the proposed literature foundation, data-readiness architecture, preprocessing pipeline, modeling suite, and evaluation protocol were formally presented. The mentor confirmed the core project formulation and provided specific technical corrections to ensure methodological rigor, zero data leakage, and alignment with Autonomous Underwater Vehicle (AUV) operational constraints.

All mentor decisions and directives have been formally incorporated into **Methodology V1** (`docs/methodology_v1.md`) and the **Updated Experiment Plan** (`experiments/updated_experiment_plan.md`).

---

## 2. Itemized Mentor Directives, Decisions & Corrections

### Directive 1: Resolution of Cell `B0007` Failure Threshold
- **Mentor Query / Feedback**: In standard benchmark literature, the lithium-ion battery failure threshold is typically 70% of nominal capacity ($1.40\text{ Ah}$ for 2.0 Ah cells). However, the original NASA cycling for cell `B0007` was halted at cycle 168 when capacity was $1.439\text{ Ah}$. How should ground truth RUL be defined for this cell?
- **Decision & Confirmation**: Formally confirm $C_{\text{EOL\_threshold}} = 1.50\text{ Ah}$ for cell `B0007` (reaching EOL at cycle 148), while retaining $1.40\text{ Ah}$ for cells `B0005`, `B0006`, and `B0018`.
- **Methodology V1 Update**: Section 1.1 explicitly formalizes this dual-threshold specification, referencing the peer-reviewed benchmark protocol established by *Qiu et al. (2024)*.

---

### Directive 2: Strict Prevention of Temporal Information Leakage
- **Mentor Query / Feedback**: Many naive machine learning implementations randomly shuffle cycles across all cells, causing catastrophic temporal leakage where the model sees future cycles of the test cell during training. What safeguards prevent this?
- **Decision & Confirmation**: Strictly enforce a **Cell-Wise Zero-Shot Split Strategy**:
  - Training Set: Cells `B0005` and `B0006`.
  - Validation Set: Cell `B0007`.
  - Out-of-Sample Test Set: Cell `B0018` (completely unseen during training and tuning).
- **Methodology V1 Update**: Section 4.1 establishes that feature normalization scalers (`MinMaxScaler`) are fit **exclusively** on training cells `B0005` and `B0006`. Test cell `B0018` is normalized out-of-sample using frozen training parameters.

---

### Directive 3: Explicit Modeling of Capacity Regeneration Rebounds
- **Mentor Query / Feedback**: Lithium-ion battery capacity curves are non-monotonic due to electrochemical relaxation during rest intervals. A standard monotonic regression model will overpredict degradation after rest periods. How is this accounted for?
- **Decision & Confirmation**: Adopt a multi-resolution signal decomposition approach (CEEMDAN / Savitzky-Golay trend-residual decoupling) to separate low-frequency thermodynamic capacity fade from high-frequency relaxation spikes.
- **Methodology V1 Update**: Section 4.2 details the signal decomposition engine, and an engineered feature `is_regeneration` ($\Delta C_k > 0.005\text{ Ah}$) is incorporated into the sliding window representation.

---

### Directive 4: Multi-Task Loss Formulation ($\mathcal{L}_{\text{RUL}} + \lambda \mathcal{L}_{\text{cap}}$)
- **Mentor Query / Feedback**: Predicting RUL purely as a scalar integer risks the neural network learning a simple linear cycle countdown rather than internal electrochemical degradation.
- **Decision & Confirmation**: Implement a **Multi-Task Neural Head** that simultaneously predicts RUL (in cycles) and next-cycle discharge capacity (in Ah), using a weighted composite loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\widehat{\text{RUL}}, \text{RUL}^*) + \lambda_{\text{cap}} \cdot \mathcal{L}_{\text{MSE}}(\hat{C}, C^*)$$
- **Methodology V1 Update**: Section 4.3 sets $\lambda_{\text{cap}} = 10.0$, confirmed through empirical validation to balance cycle scale ($0\text{--}150$) with capacity scale ($1.2\text{--}2.0\text{ Ah}$).

---

### Directive 5: Real-Time Embedded Constraints for Marine AUVs
- **Mentor Query / Feedback**: Autonomous Underwater Vehicles operate with constrained edge computing hardware (e.g., NVIDIA Jetson Orin Nano, STM32 microcontrollers, or Raspberry Pi CM4). The prognostic model must satisfy real-time onboard BMS inference budgets.
- **Decision & Confirmation**: Establish hard operational boundaries: inference latency must remain $< 5.0\text{ ms}$ per sample, and total model storage must remain $< 2.0\text{ MB}$.
- **Methodology V1 Update**: Section 5.2 benchmarks and confirms that the top-performing TCN model operates at **$1.54\text{ ms}$ latency** and **$428\text{ kB}$ memory footprint**, well within AUV onboard BMS constraints.

---

## 3. Side-by-Side Modification Traceability Table

| Mentor Feedback Area | Previous Assumption | Correction Applied in Methodology V1 | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **B0007 Failure Point** | Extrapolate B0007 to 1.40 Ah | Set B0007 EOL threshold to 1.50 Ah ($k_{\text{EOL}} = 148$) | `src/data/parser_nasa.py:L18-24` |
| **Data Split Protocol** | Random cycle train/test split | Strict Cell-Wise Zero-Shot Split (`B0005/6` $\to$ `B0007` $\to$ `B0018`) | `train_and_benchmark.py:L33-47` |
| **Feature Normalization** | Global scaling across all data | Training-only fitted `MinMaxScaler`, applied out-of-sample | `src/features/feature_builder.py:L48-56` |
| **Regeneration Handling** | Treated as sensor noise | Explicit CEEMDAN decomposition + $\Delta C_k$ indicator feature | `src/features/decomposition.py:L10-32` |
| **Objective Function** | Single-output RUL MSE Loss | Multi-Task Loss: $\mathcal{L}_{\text{RUL}} + 10.0 \cdot \mathcal{L}_{\text{capacity}}$ | `src/training/trainer.py:L58-63` |
| **Edge Hardware Budget**| No latency constraint defined | Inference latency benchmarked ($<1.6\text{ ms}$, $<1.2\text{ MB}$) | `results/metrics/model_benchmark_comparison.md` |
