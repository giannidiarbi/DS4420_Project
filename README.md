# DS4420 Project – Boston Housing Prices

This repository contains the code for our DS 4420 final project: predicting Boston median home values (`medv`) with (1) a **manual neural network** in Python and (2) **Bayesian linear regression** implemented in R via a **Gibbs sampler**.

## Methods

| Method | Language | File | Notes |
|--------|----------|------|--------|
| MLP (one hidden layer, ReLU) | Python / NumPy | `phase2_nn.py` | Manual forward/backprop (course “manual” implementation). |
| Bayesian linear regression | R (base + MASS) | `phase2_bayes.R` | Gibbs sampler; conjugate-normal prior on `beta`, inverse-gamma on `sigma^2`. |

Phase I baseline (non-Bayesian POC): `phase1_poc.py` (Random Forest via scikit-learn).

## How to run

### Python (neural network)

```bash
pip install numpy scikit-learn matplotlib
python phase2_nn.py
```

### R (Bayesian regression)

From the repository root:

```bash
Rscript phase2_bayes.R
```

### Phase I baseline

```bash
python phase1_poc.py
```

## Figures

Generated files live in `figures/`. Each is a 300 dpi PNG suitable for the project report.

| File | What it shows |
|------|----------------|
| **`mlp_train_mse.png`** | **Training loss** for the manual MLP: **full-batch mean squared error** on the **training** set at each **epoch** (horizontal axis = epoch, vertical axis = train MSE). Inputs are **standardized** using the training split before training. |
| **`mlp_pred_vs_actual.png`** | **Test-set fit** for the MLP: each point is one held-out tract; **horizontal axis = actual `medv`**, **vertical axis = predicted `medv`** (both in $1000s). The dashed line is **perfect prediction** (y = x). The title includes the **test RMSE**. |
| **`bayes_pred_vs_actual.png`** | **Test-set fit** for the **Bayesian linear** model: **actual `medv`** vs **posterior mean prediction** (Gibbs samples) on the same 80/20 split (`set.seed(42)`). The dashed line is **y = x**. The title includes the **test RMSE**. |

## Project layout

- `phase1_poc.py` — Phase I Random Forest POC.
- `phase2_nn.py` — Manual MLP training and test RMSE.
- `phase2_bayes.R` — Gibbs sampling for Bayesian linear regression; prints test RMSE/MAE and posterior summaries.
- `figures/` — PNG plots for the report.
