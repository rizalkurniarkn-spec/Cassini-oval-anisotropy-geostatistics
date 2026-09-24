# Cassini Oval Anisotropy for Geostatistical Estimation and Simulation

This repository contains the Python source code associated with the manuscript:

**A Non-Elliptical Cassini Oval Anisotropy for Geostatistical Estimation and Simulation: Application to a Laterite Nickel Deposit**

The computational workflow compares two anisotropy geometries only:

1. Elliptical anisotropy
2. Cassini Oval anisotropy

The code covers nonlinear least-squares (NLS) fitting of directional ranges, Ordinary Kriging (OK), Leave-One-Out Cross-Validation (LOOCV), Sequential Gaussian Simulation (SGS), and simulation diagnostics for limonite and saprolite thickness.

## Reported Cassini Oval parameters

To match the manuscript, the reported Cassini Oval geometry is expressed using:

- `a`: half-distance parameter associated with the two foci;
- `b`: Cassini Oval constant in the implicit equation;
- focus-axis azimuth.

The public output tables and summaries report these parameters as `a`, `b`, and azimuth. Numerical transformations used internally by the optimization/implementation are not treated as reported Cassini Oval geometry parameters.

## Repository structure

```text
cassini-oval-anisotropy-geostatistics/
├── README.md
├── CODE_INDEX.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── nls_fitting_limonite.py
│   ├── nls_fitting_saprolite.py
│   ├── nls_fitting_limonite_gaussian.py
│   ├── nls_fitting_saprolite_gaussian.py
│   ├── ordinary_kriging_limonite.py
│   ├── ordinary_kriging_saprolite.py
│   ├── sgs_limonite.py
│   └── sgs_saprolite.py
└── example/
    ├── README.md
    ├── synthetic_thickness.csv
    ├── synthetic_thickness.xlsx
    └── quick_test.py
```

## Requirements

Python 3 is required. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:

- NumPy
- pandas
- SciPy
- Matplotlib
- openpyxl

## Quick test

A synthetic dataset is provided so the repository can be tested without access to the confidential drillhole dataset.

From the repository root, run:

```bash
python example/quick_test.py
```

A successful run prints `Quick test passed.` and reports the fitted ellipse and Cassini Oval parameters used by the smoke test.

## NLS fitting

The four NLS scripts reproduce the raw-data and Gaussian-data directional-range fits reported in the manuscript:

```bash
python src/nls_fitting_limonite.py --no-show
python src/nls_fitting_saprolite.py --no-show
python src/nls_fitting_limonite_gaussian.py --no-show
python src/nls_fitting_saprolite_gaussian.py --no-show
```

Each script compares only Ellipse and Cassini Oval models and saves parameter summaries, observed-versus-predicted ranges, residuals, and figures.

## Ordinary Kriging

The OK scripts use the raw-data anisotropy parameters for limonite and saprolite. The public example reads `example/synthetic_thickness.xlsx` by default.

```bash
python src/ordinary_kriging_limonite.py
python src/ordinary_kriging_saprolite.py
```

Main configurable settings are grouped near the beginning of each script, including grid resolution, search radius, minimum/maximum neighborhood size, variogram parameters, and input columns.

Expected outputs include gridded estimates, kriging variance, model-comparison summaries, and LOOCV/diagnostic outputs where enabled.

## Sequential Gaussian Simulation

The SGS scripts use Gaussian-space variogram parameters and anisotropy geometry for each zone. They include normal-score transformation, conditional simulation, back-transformation, realization summaries, directional-variogram validation, histogram/CDF diagnostics, and accuracy-precision evaluation.

```bash
python src/sgs_limonite.py
python src/sgs_saprolite.py
```

The default research configuration uses 100 realizations. Runtime and memory requirements therefore depend on grid size, neighborhood settings, and the number of realizations.

## Inputs

The public synthetic input contains four columns:

- `X`
- `Y`
- `Limonite Thickness`
- `Saprolite Thickness`

For application to another dataset, supply equivalent coordinate and thickness fields and update the configuration section at the beginning of the relevant script if different column names are used.

## Outputs

Scripts create outputs under an `outputs/` directory when executed. Generated outputs are intentionally excluded from version control because they can be reproduced from the source code and example input.

## Data availability

The original drillhole dataset used in the study was provided by PT Vale Indonesia and is subject to confidentiality restrictions. It is therefore not distributed in this repository. The synthetic dataset contains no original drillhole coordinates or confidential measurements and is provided solely to demonstrate and test the computational workflow.

## License

This repository is released under the MIT License. See `LICENSE`.

## Contact

For scientific questions about the manuscript, please contact the corresponding author listed in the manuscript. For repository implementation questions, use the repository issue tracker or the contact information provided in the manuscript's Computer Code Availability section.
