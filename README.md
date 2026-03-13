# Regime-Aware Bayesian Deferral for Seismic Phase Picking Under Distribution Shift

ICML 2026 submission by Yasmin Akhmedova and Seraphina Korompis (Imperial College London).

## Overview

Deep-learning phase pickers such as PhaseNet achieve strong in-distribution performance but degrade silently when deployed to new seismic networks. This project introduces a regime-aware Bayesian deferral framework that detects when PhaseNet's predictions are unreliable and routes those picks to a human analyst. An HMM classifies the seismic environment into *Quiet*, *Active*, and *Decaying* regimes; a Beta-Bernoulli model estimates PhaseNet's reliability in each regime; and a combined deferral score merges both signals with PhaseNet's own confidence.

The framework is evaluated on the Kazakhstan seismic network (out-of-distribution) against a Northern California baseline (in-distribution).

## Repository Structure

```
.
├── Notebook - 3 March.ipynb        # Main analysis notebook (data, models, evaluation)
├── paper_figures.ipynb             # Publication-quality figure generation
├── Sandbox.ipynb                   # Exploratory analysis
├── requirements.txt                # Python dependencies
├── kz_results_df.csv               # Kazakhstan evaluation results
├── results_df.csv                  # California evaluation results
├── paper/                          # LaTeX source (ICML 2026 format)
│   ├── main.tex
│   ├── paper.bib
│   └── sections/                   # Modular paper sections
└── figures/                        # Generated figures (from paper_figures.ipynb)
```

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: ObsPy, SeisBench, PyTorch, NumPy, Pandas, SciPy, Matplotlib, Cartopy, scikit-learn.

## Reproducing Results

1. Run `Notebook - 3 March.ipynb` end-to-end to download waveforms, run PhaseNet, fit the HMM, and evaluate deferral strategies.
2. Run `paper_figures.ipynb` to generate publication figures from the saved result DataFrames.

Note: waveform downloads require internet access and may take time depending on FDSN server availability.
