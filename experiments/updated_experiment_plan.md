# Updated Experiment Plan: Deep Learning Battery Prognostics

**Programme**: TIH, IIT Guwahati — 4-Week Online Internship Project  
**Track**: Group O4 — Deep Learning / Battery Prognostics / Autonomous Systems  
**Intern**: Rajnish Singh  
**Milestone**: Day 6 — Pre-Dataset Readiness + Mentor Review  
**Status**: Approved & Updated Post-Mentor Review  

---

## 1. Experiment Matrix & Model Configurations

The experiment suite systematically compares **8 prognostic methodologies** under standardized, leakage-free data splits:

| Exp ID | Model Architecture | Core Mechanism | Parameter Count | Primary Hyperparameters | Loss Function |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **EXP-01** | Empirical Double-Exponential | Physical curve fitting: $C(k) = ae^{bk} + ce^{dk}$ | 4 | Nonlinear Least Squares (`curve_fit`) | Residual Sum of Squares |
| **EXP-02** | Random Forest Regressor | Ensemble bagging of 100 decision trees | ~50,000 | `n_estimators=100`, `max_depth=None` | MSE Criterion |
| **EXP-03** | Standard LSTM | 2-Layer Recurrent Neural Network | 52,610 | `hidden_dim=64`, `dropout=0.2` | Multi-Task MSE |
| **EXP-04** | Standard GRU | 2-Layer Gated Recurrent Unit | 39,490 | `hidden_dim=64`, `dropout=0.2` | Multi-Task MSE |
| **EXP-05** | Temporal Convolutional Network (TCN) | Dilated Causal 1D Convolutions + ResBlocks | 107,554 | `channels=[32,64,128]`, `k=3`, `d=[1,2,4]` | Multi-Task MSE |
| **EXP-06** | BiLSTM + Multi-Head Attention | Bidirectional LSTM + Self-Attention | 220,674 | `hidden_dim=64`, `heads=4`, `dropout=0.2` | Multi-Task MSE |
| **EXP-07** | Temporal Transformer | Positional Encoding + Multi-Head Encoder | 67,714 | `d_model=64`, `heads=4`, `dim_ff=128` | Multi-Task MSE |
| **EXP-08** | Proposed Hybrid SOTA | CEEMDAN + TCN (HF) + BiLSTM (LF) + DualAttn | 293,090 | `hidden_dim=64`, `tcn_ch=[32,64]`, `h=4` | Multi-Task MSE |

---

## 2. Standardized Training & Optimization Protocol

To ensure 100% fair and reproducible benchmarking across all models:
- **Compute Hardware**: Apple Silicon MPS (Metal Performance Shaders) / CUDA / CPU auto-detection via `src/utils/helpers.py`.
- **Random Seed**: Fixed globally to `seed = 42` across Python `random`, `numpy`, and `torch`.
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\text{weight\_decay} = 10^{-4}$).
- **Learning Rate**: Initial $\eta_0 = 10^{-3}$ with adaptive decay:
  $$\eta_{t+1} = 0.5 \cdot \eta_t \quad \text{if val loss plateaus for 5 epochs}$$
- **Batch Size**: 16 sliding sequence samples per batch.
- **Maximum Epochs**: 80 epochs with EarlyStopping (patience = 15 epochs on validation loss).
- **Gradient Clipping**: Maximum norm $\Vert \mathbf{g} \Vert_2 \le 1.0$ to prevent gradient explosion.
- **Multi-Task Loss Weight**: $\lambda_{\text{cap}} = 10.0$ balancing cycle error with capacity Ah scale:
  $$\mathcal{L} = \text{MSE}(\widehat{\text{RUL}}, \text{RUL}^*) + 10.0 \cdot \text{MSE}(\hat{C}, C^*)$$

---

## 3. Planned Ablation Studies (Weeks 2–3)

### Ablation Study 1: Impact of Temporal Sliding Window Length ($L$)
- **Objective**: Determine the optimal sequence memory horizon for capturing degradation trajectory without excess computational overhead.
- **Levels Tested**: $L \in \{5, 10, 15, 20, 25\text{ cycles}\}$.
- **Hypothesis**: $L = 15$ provides the ideal trade-off between capturing multi-cycle capacity regeneration rebounds and minimizing sequence latency.

### Ablation Study 2: Multi-Task Loss Capacity Regularization ($\lambda_{\text{cap}}$)
- **Objective**: Quantify the impact of joint SOH capacity estimation on RUL prediction accuracy.
- **Levels Tested**: $\lambda_{\text{cap}} \in \{0.0, 1.0, 5.0, 10.0, 20.0, 50.0\}$.
- **Hypothesis**: Constraining the latent space with physical capacity prediction prevents the network from overfitting to linear cycle counters.

### Ablation Study 3: Signal Decomposition Efficacy (CEEMDAN vs Raw)
- **Objective**: Measure performance gains from separating high-frequency regeneration from low-frequency thermodynamic fade.
- **Variants Tested**:
  1. Raw input telemetry without decomposition.
  2. Savitzky-Golay / Trend-Residual decoupled inputs.
  3. Full CEEMDAN multi-IMF decomposition with sample entropy modal classification.

---

## 4. Cross-Cell Robustness & Generalization Plan

To prove that models do not simply memorize specific cell trajectories, the following cross-cell evaluation matrices are scheduled:

1. **Fold A (Primary Benchmark)**: Train on `B0005` + `B0006`, Val on `B0007`, Test on `B0018`.
2. **Fold B (Leave-B0005-Out)**: Train on `B0006` + `B0007`, Val on `B0018`, Test on `B0005`.
3. **Fold C (Leave-B0006-Out)**: Train on `B0005` + `B0007`, Val on `B0018`, Test on `B0006`.
4. **Fold D (Marine AUV Domain Transfer)**: Train on all NASA cells (`B0005`–`B0018`), Zero-Shot Test on simulated AUV mission profile with thermal bathymetric fluctuations ($4^\circ\text{C}$ to $20^\circ\text{C}$).

---

## 5. Artifact & Experiment Logging Framework

Every experiment run automatically registers and saves:
1. **Model Checkpoint**: Saved to `saved_models/<model_name>_best.pt`.
2. **Tabular Metrics**: Appended to `results/metrics/model_benchmark_comparison.csv`.
3. **Markdown Comparison Table**: Generated in `results/metrics/model_benchmark_comparison.md`.
4. **Degradation Stage Breakdown**: Saved to `results/metrics/degradation_stages_robustness.csv`.
5. **Visualization Artifacts**: Saved at 300 DPI in `results/figures/`.
6. **Execution Log**: Streamed to `experiments/training.log`.
