# Battery Health Prognostics & Remaining Useful Life (RUL) Prediction for Autonomous Underwater Vehicles (AUVs)

[![TIH IIT Guwahati](https://img.shields.io/badge/TIH%20IIT%20Guwahati-Internship%20Phase-00529B.svg)](https://tih.iitg.ac.in)
[![Track](https://img.shields.io/badge/Track-Group%20O4-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20PCoE-orange.svg)](https://data.nasa.gov/)

An end-to-end, production-grade deep learning framework for **Battery State of Health (SOH)** estimation and **Remaining Useful Life (RUL)** forecasting under marine mission profiles and dynamic degradation dynamics.

Developed as part of the **TIH IIT Guwahati 4-Week Online Internship Research Project** (Group O4).

---

## 📌 Submission Milestones & Review Deliverables

| Milestone / Stage | Deliverable Document | Key Content & Inclusions |
| :--- | :--- | :--- |
| **Pre-Dataset Readiness & Mentor Review (Day 6)** | [Methodology Document V1](docs/methodology_v1.md) | **Methodology V1**: Mathematical SOH & RUL definitions, literature review (*Qiu et al., 2024*, *Ma et al., 2025*), data-readiness package (34 NASA cells), CEEMDAN decomposition, 8-model architecture zoo, and multi-task loss ($\mathcal{L}_{\text{RUL}} + 10.0 \cdot \mathcal{L}_{\text{cap}}$). |
| **Mentor Decision & Corrections Log** | [Mentor Review Corrections](reports/mentor_review_corrections.md) | **Mentor Corrections**: Confirmed EOL failure thresholds ($1.40\text{ Ah}$ for B0005/6/18, $1.50\text{ Ah}$ for B0007), strict cell-wise zero-leakage split, out-of-sample scaling, capacity regeneration modeling, and AUV embedded edge constraints. |
| **Updated Experiment Plan** | [Updated Experiment Plan](experiments/updated_experiment_plan.md) | **Experiment Plan V1**: 8-model comparison matrix, hyperparameter search grids, AdamW optimizer schedule, sequence length ($L=15$) ablation plan, and cross-cell generalization folds. |
| **Mentor Review Presentation** | [3-Slide Review Pack](reports/mentor_review_2_presentation_pack.md) | **Mentor Review 2 Pack**: Slide 1 (Pre-Dataset Readiness), Slide 2 (Preprocessing, Modeling & Evaluation Plan), Slide 3 (Mentor Corrections & Validated Constraints). |
| **Week 2 — Day 1** | [Problem Understanding](docs/week2_day1_problem_understanding_and_system_definition.md) | Technical problem formulation, mathematical SOH/RUL definitions, end-to-end software pipeline diagram, and 8+ mentor review questions. |
| **Week 2 — Day 2** | [Data Exploration & Baseline](docs/week2_day2_data_exploration_and_baseline_experiment.md) | Quality audit of 34 NASA cells (2,744 cycles), leakage-safe split, empirical baseline vs Random Forest vs 6 deep learning models, challenges & findings. |
| **Interactive Submission Notebook** | [Week 2 Submission Notebook](notebooks/week2_day1_day2_submission.ipynb) | Executable Jupyter walkthrough of data inspection, baseline experiments, and benchmark visualization. |

---

## 🌟 Key Highlights & Innovations

- **Domain-Specific Formulation**: Formulated for Autonomous Underwater Vehicles (AUVs) and marine energy storage systems facing thermal gradients and pulsed load surges (*Ma et al., 2025*).
- **Novel Hybrid Architecture**: Proposed **CEEMDAN-TCN-BiLSTM-DualAttention** network combining multi-scale dilated causal temporal convolutions, squeeze-and-excitation channel attention, bidirectional recurrent units, and multi-head self-attention (*Qiu et al., 2024*).
- **Capacity Regeneration Modeling**: Explicitly models non-linear electrochemical capacity rebound during resting intervals.
- **Strict Leakage Prevention**: Full zero-shot cross-cell validation (Trained on `B0005` & `B0006`, validated on `B0007`, tested out-of-sample on `B0018`).
- **Comprehensive Benchmarks**: Compares 8 distinct prognostic algorithms across MAE, RMSE, MAPE, $R^2$, training duration, inference latency, and model size.

---

## 📂 Repository Structure

```tree
├── data/
│   ├── raw/                # 34 NASA .mat battery aging files
│   └── processed/          # Cleaned cycle-by-cycle tabular CSVs (B0005-B0056)
├── docs/
│   ├── methodology_v1.md   # Comprehensive Methodology V1 (Approved)
│   ├── mentor_review_corrections.md # Mentor feedback and decision log
│   ├── week2_day1_problem_understanding_and_system_definition.md
│   └── week2_day2_data_exploration_and_baseline_experiment.md
├── experiments/
│   ├── updated_experiment_plan.md   # Detailed experiment matrix & ablation plans
│   └── training.log        # Hardware execution & training logs
├── notebooks/
│   ├── week2_day1_day2_submission.ipynb
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training_and_benchmarking.ipynb
│   └── 03_robustness_and_cross_cell_evaluation.ipynb
├── reports/
│   ├── mentor_review_2_presentation_pack.md  # 3-Slide Mentor Review 2 Pack
│   ├── mentor_review_corrections.md          # Itemized feedback traceability table
│   ├── week2_mentor_review_progress_pack.md  # Week 2 Day 1-2 Progress Pack
│   └── final_research_report.md              # Comprehensive research report
├── results/
│   ├── figures/            # High-resolution publication-quality plots (300 DPI)
│   └── metrics/            # CSV and Markdown benchmark comparison tables
├── saved_models/           # Serialized PyTorch model checkpoints (.pt)
├── src/
│   ├── data/               # NASA/CALCE parsers and AUV mission simulator
│   ├── features/           # Signal decomposition & sliding window builders
│   ├── models/             # PyTorch architectures (LSTM, GRU, TCN, BiLSTM-Attn, Transformer, Hybrid)
│   ├── training/           # PyTorch multi-task training engine with EarlyStopping
│   ├── evaluation/         # Metrics, degradation-stage robustness & failure analysis
│   └── utils/              # Seed control, hardware accelerator detection, loggers
├── eda.py                  # Automated Exploratory Data Analysis pipeline
├── train_and_benchmark.py  # End-to-end model training & evaluation pipeline
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/givemehat/battery-health-rul-deep-learning.git
cd battery-health-rul-deep-learning

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Parses raw NASA battery files, extracts cycle telemetry, and generates publication figures in `results/figures/`:
```bash
python eda.py
```

### 3. Train & Benchmark All Models
Trains all 8 architectures, evaluates cross-cell generalizability on unseen battery `B0018`, and outputs benchmark comparison tables:
```bash
python train_and_benchmark.py
```

---

## 📊 Benchmark Results

Evaluated on unseen test battery cell **NASA B0018** (Trained on `B0005`, `B0006`; Validated on `B0007`):

| Model Architecture | MAE (Cycles) | RMSE (Cycles) | MAPE (%) | $R^2$ Score | Max Error | Params | Train Time (s) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Convolutional Network (TCN)** | **5.050** | **6.049** | **41.27%** | **0.952** | **13.08** | **107,554** | **5.20s** | **1.54 ms** |
| **Random Forest Regressor** | **7.100** | **7.947** | **43.62%** | **0.916** | **18.21** | 50,000 | 0.36s | 0.05 ms |
| **Proposed Hybrid SOTA (CEEMDAN-TCN-BiLSTM-DualAttn)** | 14.397 | 15.809 | 109.87% | 0.669 | 22.05 | 293,090 | 4.96s | 0.28 ms |
| **Temporal Transformer** | 14.507 | 16.041 | 115.68% | 0.659 | 23.32 | 67,714 | 11.39s | 1.44 ms |
| **Empirical Double-Exp Baseline** | 17.432 | 19.196 | 133.88% | 0.512 | 22.00 | 4 | 0.01s | 0.10 ms |
| **BiLSTM-Attention** | 17.536 | 20.209 | 142.11% | 0.459 | 30.94 | 220,674 | 9.41s | 1.35 ms |
| **Standard LSTM** | 20.637 | 22.721 | 158.05% | 0.316 | 31.41 | 52,610 | 11.30s | 1.12 ms |
| **Standard GRU** | 21.666 | 23.597 | 159.93% | 0.262 | 32.30 | 39,490 | 22.64s | 0.35 ms |

---

## 📈 Visual Results

### 1. Degradation Curves & Capacity Regeneration
![Degradation Curves](results/figures/fig1_capacity_degradation_curves.png)
*Figure 1: Discharge capacity degradation curves across NASA cells & simulated AUV mission profile.*

![Regeneration Zoom](results/figures/fig2_capacity_regeneration_zoom.png)
*Figure 2: Non-monotonic capacity recovery spikes observed after chemical rest intervals.*

### 2. Multi-Resolution Signal Decomposition
![Signal Decomposition](results/figures/fig5_signal_decomposition.png)
*Figure 3: CEEMDAN-inspired decomposition isolating thermodynamic trend vs regeneration noise.*

### 3. Model Trajectory Predictions & Residual Errors
![Model Predictions](results/figures/fig6_model_rul_predictions_b0018.png)
*Figure 4: Remaining Useful Life forecasting trajectory across all models on test cell B0018.*

![Residual Distribution](results/figures/fig7_residual_error_distribution.png)
*Figure 5: Residual prediction error distribution within the high-confidence ±5 cycle envelope.*

---

## 🔬 References & Acknowledgments

1. **Qiu, S., Zhang, B., Lv, Y., Zhang, J., & Zhang, C. (2024)**. *A Lithium-Ion Battery Remaining Useful Life Prediction Model Based on CEEMDAN Data Preprocessing and HSSA-LSTM-TCN*. World Electric Vehicle Journal, 15(5), 177.
2. **Ma, D., Li, Y., Ma, T., & Pascoal, A. M. (2025)**. *The state of the art in key technologies for autonomous underwater vehicles: a review*. Ocean Engineering Review / Engineering.
3. **NASA Prognostics Center of Excellence (PCoE)**. *Li-ion Battery Aging Dataset*, NASA Ames Research Center.
4. **TIH, IIT Guwahati**: 4-Week Online Internship Research Program in Deep Learning and Battery Prognostics.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
