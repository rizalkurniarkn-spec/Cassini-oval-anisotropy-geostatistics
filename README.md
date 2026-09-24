# Cassini Oval Anisotropy for Geostatistical Estimation and Simulation

This repository contains the source code associated with the manuscript:

**"A Non-Elliptical Cassini Oval Anisotropy for Geostatistical Estimation and Simulation: Application to a Laterite Nickel Deposit"**

The repository provides the computational implementation used to investigate conventional elliptical anisotropy and the proposed non-elliptical Cassini Oval anisotropy model for geostatistical estimation and simulation of laterite zone thickness.

## Overview

The computational workflow includes:

- directional experimental variogram analysis;
- elliptical anisotropy modeling;
- Cassini Oval anisotropy modeling;
- effective-distance calculation;
- nonlinear least-squares (NLS) fitting;
- Ordinary Kriging (OK);
- Leave-One-Out Cross-Validation (LOOCV);
- Sequential Gaussian Simulation (SGS);
- statistical evaluation and visualization of estimation and simulation results.

The methodology was applied to the thickness of limonite and saprolite zones in a laterite nickel deposit.

## Repository Structure

The repository is organized as follows:

```text
cassini-oval-anisotropy-geostatistics/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── directional_variogram.py
│   ├── ellipse_anisotropy.py
│   ├── cassini_anisotropy.py
│   ├── nls_fitting.py
│   ├── ordinary_kriging.py
│   ├── loocv.py
│   └── sequential_gaussian_simulation.py
│
├── example/
│   ├── synthetic_data.csv
│   └── run_example.py
│
└── outputs/
    └── example_results/
