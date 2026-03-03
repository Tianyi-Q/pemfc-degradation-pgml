# **Physics-Guided Machine Learning (PGML) for Electrochemical Degradation**

[![Physics-Informed Gradient Test](https://github.com/Tianyi-Q/pemfc-degradation-pgml/actions/workflows/physics_test.yml/badge.svg)](https://github.com/Tianyi-Q/pemfc-degradation-pgml/actions/workflows/physics_test.yml)

An automated pipeline and Physics-Informed Neural Network (PINN) architecture designed to ingest, process, and model multivariable fuel cell degradation data. 

## Architecture & Physics-Informed Loss

This PGNN incorporates a custom differential loss function. By leveraging PyTorch's `autograd` engine, the model dynamically computes $\frac{\partial V}{\partial t}$ during the forward pass.

The total loss manifold is defined as:

$$
\mathcal{L}_{total} = \text{MSE}(V_{pred}, V_{true}) + \lambda \frac{1}{N} \sum \text{ReLU}\left(\frac{\partial V_{pred}}{\partial t}\right)
$$

This constraint forces the latent space to respect the irreversible kinetics of catalyst degradation, actively rejecting transient sensor noise and flooding artifacts.

## Current Testing Baseline (Synthetic Matrix)

Pending raw experimental data ingestion, the pipeline is currently validated against a high-fidelity synthetic baseline mirroring a 4x4 matrix over 500 hours:

* **TiO2 Loadings:** 0.05, 0.1, 0.2, 0.3
* **Relative Humidity:** 30%, 60%, 80%, 100%

## Environment Setup

Optimized for CUDA 12.1 and Python 3.12.

```bash
git clone [https://github.com/Tianyi-Q/pemfc-degradation-pgml.git](https://github.com/Tianyi-Q/pemfc-degradation-pgml.git)
cd pemfc-degradation-pgml
pip install -r requirements.txt
python src/generate_matrix.py
python src/train.py
```
