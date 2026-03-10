# Physics-Guided Machine Learning for Fuel-Cell Degradation

[![PEMFC-PGML](https://github.com/Tianyi-Q/pemfc-degradation-pgml/actions/workflows/physics_test.yml/badge.svg)](https://github.com/Tianyi-Q/pemfc-degradation-pgml/actions/workflows/physics_test.yml)

A small end-to-end pipeline that **predicts how a hydrogen fuel cell's voltage degrades over time**, using a neural network whose training is guided by electrochemical physics.

> **Hobby project disclaimer** — this is my first time working with neural networks. After getting frustrated with a purely data-driven model that flatlines like my heart at 8 o'clock in the morning class, I decided to hammer physics directly into the loss function so the model can't cheat. It is simplified and I might update it later.

---

## What is this about?

A **PEM fuel cell** (Proton-Exchange Membrane Fuel Cell) converts hydrogen and oxygen into electricity. Over hundreds of hours of operation the cell slowly **degrades** — its voltage drops because the catalyst dissolves, the membrane dries out or floods, and internal resistance rises.

Predicting *how fast* that degradation happens under different operating conditions is valuable for maintenance planning and lifetime estimation. This project does that with a **Physics-Guided Neural Network (PGNN)**: a standard neural network plus an extra training penalty that encodes a physical rule — *voltage should not recover faster than is physically possible*.

### Why "physics-guided"?

A normal neural network minimises only data error. It is free to learn patterns that violate physics, for example predicting that a degrading cell suddenly gains voltage out of nowhere. Adding a **physics loss** term (a soft constraint on the voltage derivative dV/dt) prevents that. The total training objective is:

```
Total Loss = Data Loss + λ × Physics Loss
```

where **Data Loss** measures prediction accuracy and **Physics Loss** penalises unrealistically fast voltage recovery.

---

## Pipeline overview

The project runs as a **three-command** pipeline. You only run the scripts listed below — the helper modules (`data_loader.py`, `model.py`) are imported automatically and never need to be executed directly.

| Step | Script you run         | What it does                                                                                                                                                                                                |
| ---- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `generate_matrix.py` | Creates synthetic but physically motivated fuel-cell data (Nernst voltage, activation/ohmic/concentration losses, five dynamic states, AR(1) noise) over a 4×4 condition matrix.                           |
| 2    | `train.py`           | Trains a compact MLP (≈17 000 parameters) with SmoothL1 data loss + dV/dt physics penalty, Adam optimiser, LR scheduling, and early stopping. Saves a checkpoint with model weights and scaler statistics. |
| 3    | `evaluate.py`        | Loads the checkpoint, runs inference on every operating segment, computes error metrics (MAE, RMSE, R²), and generates three plots + a metrics CSV.                                                        |

| Helper module (imported, not run) | Role                                                                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_loader.py`                | Reads the CSV, normalises column names, selects features, and scales everything to zero-mean/unit-variance. Imported by `train.py` and `evaluate.py`. |
| `model.py`                      | Defines the PGNN neural network architecture. Imported by `train.py` and `evaluate.py`.                                                               |

```
generate_matrix.py ──→ train.py ──→ evaluate.py
       (data)          (training)     (results)
                          ↑               ↑
                   data_loader.py    data_loader.py
                   model.py         model.py
                   (imported)        (imported)
```

### Model architecture

A 4-layer fully connected network: **input → 64 → 64 → 64 → 1**, with Tanh activations (chosen because they are smooth everywhere, which the physics-loss derivative computation requires).

---

## Current results (synthetic baseline)

Because real degradation data requires hundreds of hours of lab time, the pipeline is currently validated on **synthetic data** that mimics a 4×4 experiment matrix:

| Condition             | Values                  |
| --------------------- | ----------------------- |
| TiO2 catalyst loading | 0.05, 0.10, 0.20, 0.30  |
| Relative humidity     | 30 %, 60 %, 80 %, 100 % |
| Duration              | 500 hours per segment   |

These conditions mirror a previous experimental study — for more info visit [tianyi-q.github.io](https://tianyi-q.github.io/).

**Global metrics on the full dataset:**

| MAE       | RMSE      | R²    |
| --------- | --------- | ------ |
| 0.00160 V | 0.00200 V | 0.9877 |

---

## Quick start

**Requirements:** Python 3.12, CUDA 12.1 (CPU also works, just slower). Other versions may work but are untested.

```bash
git clone https://github.com/Tianyi-Q/pemfc-degradation-pgml.git
cd pemfc-degradation-pgml
pip install -r requirements.txt
```

Then run the full pipeline:

```bash
python src/generate_matrix.py   # 1. create synthetic dataset
python src/train.py             # 2. train the PGNN
python src/evaluate.py          # 3. evaluate and generate plots
```

Outputs land in:

| Artifact                 | Path                                                    |
| ------------------------ | ------------------------------------------------------- |
| Synthetic data CSV       | `data/raw/synthetic_matrix.csv`                       |
| Trained model checkpoint | `models/pgnn_checkpoint.pth`                          |
| Multi-segment plot       | `data/processed/pgnn_validation_multisegment.png`     |
| Parity plot              | `data/processed/pgnn_validation_parity.png`           |
| Local comparison plot    | `data/processed/pgnn_validation_local_comparison.png` |
| Per-segment metrics CSV  | `data/processed/pgnn_validation_metrics.csv`          |

---

## Project structure

```
pemfc-degradation-pgml/
├── src/
│   ├── generate_matrix.py    # Synthetic data generator (step 1)
│   ├── data_loader.py        # Dataset class & schema normalisation (imported by train/evaluate)
│   ├── model.py              # PGNN architecture definition (imported by train/evaluate)
│   ├── train.py              # Training loop with physics loss (step 2)
│   └── evaluate.py           # Evaluation, metrics & plots (step 3)
├── old/                      # Earlier model version (see note below)
├── data/
│   ├── raw/                  # Generated synthetic CSV + plot
│   └── processed/            # Evaluation outputs
├── models/                   # Saved weights & checkpoint
├── requirements.txt
└── README.md
```

> **Note:** the `old/` folder contains a previous (simpler) version of the model. To run those scripts, copy them into `src/` so the relative file paths work.

---

## What the physics model simulates

The synthetic data generator (`generate_matrix.py`) is not a random-number machine — it implements simplified but physically grounded electrochemistry:

- **Nernst equation** — thermodynamic open-circuit voltage as a function of temperature and gas pressures.
- **Activation loss** — Tafel kinetics for the sluggish oxygen reduction reaction.
- **Ohmic loss** — proton conductivity through the Nafion membrane (depends on hydration and temperature).
- **Concentration loss** — mass-transport limitation as current approaches limiting current (worsened by flooding).
- **Five dynamic states** evolved over time: membrane hydration, flooding, cell temperature, catalyst surface-area decay, and correlated sensor noise.

All coefficients are hand-tuned placeholders. With real experimental data, you would calibrate them to match measured polarisation curves and durability trends.

---

## Dependencies

| Package      | Version     | Role                                     |
| ------------ | ----------- | ---------------------------------------- |
| PyTorch      | 2.4.1+cu121 | Neural network training & autograd       |
| pandas       | 2.2.3       | CSV reading & data manipulation          |
| NumPy        | 2.2.6       | Numerical computation                    |
| scikit-learn | 1.5.2       | StandardScaler for feature normalisation |
| matplotlib   | 3.8.4       | Plot generation                          |

---

## CI

A GitHub Actions workflow (`.github/workflows/physics_test.yml`) runs the full pipeline on every push/PR to `main` — generate data, train, evaluate — to make sure nothing is broken.
