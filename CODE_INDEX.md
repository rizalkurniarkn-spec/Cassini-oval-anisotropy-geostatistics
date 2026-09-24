# Code index

This repository contains six research scripts organized by workflow and laterite zone. Only the **Ellipse** and **Cassini Oval** anisotropy models are included in the model comparisons.

| File | Purpose | Public example input |
|---|---|---|
| `src/nls_fitting_limonite.py` | NLS fitting of elliptical and affine-scaled Cassini Oval directional ranges for limonite | Directional ranges embedded in the script |
| `src/nls_fitting_saprolite.py` | NLS fitting of elliptical and affine-scaled Cassini Oval directional ranges for saprolite | Directional ranges embedded in the script |
| `src/ordinary_kriging_limonite.py` | Ordinary Kriging comparison: Ellipse vs Cassini Oval for limonite | `example/synthetic_thickness.xlsx` |
| `src/ordinary_kriging_saprolite.py` | Ordinary Kriging comparison: Ellipse vs Cassini Oval for saprolite, including diagnostics/LOOCV available in the supplied workflow | `example/synthetic_thickness.xlsx` |
| `src/sgs_limonite.py` | Sequential Gaussian Simulation comparison: Ellipse vs Cassini Oval for limonite | `example/synthetic_thickness.xlsx` |
| `src/sgs_saprolite.py` | Sequential Gaussian Simulation comparison: Ellipse vs Cassini Oval for saprolite | `example/synthetic_thickness.xlsx` |

## Important data note

The Excel/CSV file in `example/` is synthetic and contains no original exploration data. The original drillhole dataset is not included because it is subject to confidentiality restrictions.
