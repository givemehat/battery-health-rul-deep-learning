# TIH, IIT Guwahati | AI/ML Online Internship Programme
## 3-Slide Progress Pack — Mentor Review (Week 2, Day 1 & Day 2)

**Track**: Group O4 — Deep Learning-Based Battery Health Monitoring and Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles  
**Intern**: Rajnish Singh  
**Date**: Week 2 (Day 1 & Day 2 Milestones)  

---

### Slide 1: Work Completed (Week 2, Day 1 & Day 2)
1. **Problem Formulation & Mathematical Definition**:
   - Formulated SOH ($\text{SOH}_k = \frac{C_k}{C_{\text{nominal}}} \times 100\%$) and RUL ($\text{RUL}_k = \max(0, k_{\text{EOL}} - k)$) with standardized EOL thresholds ($1.40\text{ Ah}$ for B0005/B0006/B0018, $1.50\text{ Ah}$ for B0007).
   - Mapped domain requirements to marine Autonomous Underwater Vehicle (AUV) energy systems and battery thermal dynamics (*Ma et al., 2025*).
2. **Dataset Readiness & Quality Audit**:
   - Ingested and parsed all 34 NASA PCoE battery aging `.mat` files (**2,744 total discharge cycles**).
   - Built an automated cycle segmentation pipeline extracting voltage, current, surface temperature, energy throughput, and capacity regeneration indicators.
   - Designed a simulated AUV marine mission profile with variable thermal loads ($4^\circ\text{C}$ to $20^\circ\text{C}$) and pulsed thruster power surges.
3. **Leakage-Safe Temporal Pipeline & Baseline Experiments**:
   - Implemented cell-wise zero-leakage temporal split: Train on `B0005` & `B0006` (308 samples), Val on `B0007` (154 samples), Test out-of-sample on `B0018` (118 samples).
   - Built and trained 8 prognostic architectures: Empirical Double-Exponential Baseline, Random Forest, LSTM, GRU, TCN, BiLSTM-Attention, Temporal Transformer, and Proposed Hybrid SOTA.

---

### Slide 2: Results & Key Findings
1. **Comparative Model Performance on Unseen Test Cell (`B0018`)**:
   - **TCN (Temporal Convolutional Network)** achieved top performance with **MAE = 5.05 cycles**, **RMSE = 6.05 cycles**, and **$R^2 = 0.952$**, utilizing dilated causal 1D convolutions to capture localized capacity regeneration without vanishing gradients.
   - **Random Forest Regressor** achieved strong baseline accuracy with **MAE = 7.10 cycles** ($R^2 = 0.916$).
   - Classical empirical baseline failed under non-monotonic capacity rebound (**MAE = 17.43 cycles**).
2. **Degradation-Stage Robustness**:
   - **Early-Stage (Cycles 1–60)**: MAE = **2.73 cycles** ($R^2 = 0.886$).
   - **Mid-Stage (Cycles 61–120)**: MAE = **6.86 cycles** (impacted by regeneration shocks).
   - **Late-Stage / Near-EOL (Cycles > 120)**: MAE = **4.90 cycles**, Max Error tightly bounded at **7.02 cycles**.
3. **Computational Benchmark**:
   - Inference latency across all neural models remained under **$<1.6\text{ ms}$** per cycle on Apple MPS/CPU, satisfying real-time AUV BMS requirements.

---

### Slide 3: Blockers & Next 3-Day Plan
1. **Current Blockers / Technical Challenges**:
   - *Cross-Cell Initial Capacity Offset*: Cell initial capacities vary ($1.85\text{ Ah}$ for B0005 vs $2.03\text{ Ah}$ for B0006), requiring normalized relative feature representations.
   - *Hyperparameter Balancing in Hybrid SOTA*: CEEMDAN sub-branch fusion requires adaptive loss weighting to avoid underfitting high-frequency modes.
2. **Next 3-Day Action Plan (Week 2, Day 3 to Day 5)**:
   - **Day 3 (Feature Engineering & Decomposition Optimization)**: Fine-tune CEEMDAN and multi-scale wavelet decomposition modes to decouple thermodynamic aging from rest regeneration.
   - **Day 4 (Multi-Step Trajectory Rollout)**: Expand single-step RUL predictions to recursive multi-step forecasting ($k \to k+30$ horizon).
   - **Day 5 (Ablation Study & Physics-Informed Regularization)**: Conduct formal ablation on attention heads, causal dilation factors, and physics-informed thermodynamic loss bounds.
