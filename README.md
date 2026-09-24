# Cassini Oval Anisotropy for Geostatistical Estimation and Simulation

This repository contains the source code associated with the manuscript:

**"A Non-Elliptical Cassini Oval Anisotropy for Geostatistical Estimation and Simulation: Application to a Laterite Nickel Deposit"**

The repository provides the computational implementation used to compare conventional elliptical anisotropy with the proposed non-elliptical Cassini Oval anisotropy model for geostatistical estimation and simulation of laterite zone thickness.

## Overview

The computational workflow includes:

- directional anisotropy modeling;
- elliptical anisotropy fitting;
- Cassini Oval anisotropy fitting;
- nonlinear least-squares (NLS) optimization;
- directional range evaluation;
- Ordinary Kriging (OK);
- Leave-One-Out Cross-Validation (LOOCV);
- Sequential Gaussian Simulation (SGS);
- normal-score transformation and back-transformation;
- statistical evaluation and uncertainty analysis.

The methodology is implemented separately for the limonite and saprolite zones of a laterite nickel deposit.

The main comparison throughout the repository is between:

1. **Elliptical anisotropy**, representing the conventional anisotropy model; and
2. **Cassini Oval anisotropy**, representing the proposed non-elliptical spatial continuity model.

## Repository Structure

```text
cassini-oval-anisotropy-geostatistics/
│
├── README.md
├── CODE_INDEX.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── nls_fitting_limonite.py
│   ├── nls_fitting_saprolite.py
│   ├── ordinary_kriging_limonite.py
│   ├── ordinary_kriging_saprolite.py
│   ├── sgs_limonite.py
│   └── sgs_saprolite.py
│
└── example/
    ├── README.md
    ├── synthetic_thickness.csv
    ├── synthetic_thickness.xlsx
    └── quick_test.py
