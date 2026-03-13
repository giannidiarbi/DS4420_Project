# DS4420 Project – Boston Housing Prices

This repository contains the code and artifacts for our DS 4420 final project, which focuses on predicting Boston housing prices using machine learning methods, including neural networks and Bayesian approaches.

## Phase I Contents

- Literature review (`Literary Review - DS4420.pdf`), including a link to this GitHub repository.
- A simple **non‑Bayesian proof‑of‑concept model** implemented in Python in `phase1_poc.py`.

The Phase I model:

- Uses the Boston Housing dataset from OpenML.
- Trains a basic `RandomForestRegressor` and reports test RMSE.
- Serves only as an initial baseline; more sophisticated models and a Bayesian method will be added for Phase II.

## How to run the Phase I proof‑of‑concept model

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install numpy scikit-learn
   ```

3. Run the script from the repository root:

   ```bash
   python phase1_poc.py
   ```

You should see the model train and a test RMSE printed to the console.
