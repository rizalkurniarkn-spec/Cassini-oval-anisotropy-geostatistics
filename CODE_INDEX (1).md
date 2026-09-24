# Code index

The repository contains eight research scripts. Model comparisons include only **Ellipse** and **Cassini Oval** anisotropy.

| File | Data scale | Purpose | Public example input |
|---|---|---|---|
| `src/nls_fitting_limonite.py` | Raw | NLS fitting of Ellipse and Cassini Oval to limonite directional ranges | Directional ranges embedded in the script |
| `src/nls_fitting_saprolite.py` | Raw | NLS fitting of Ellipse and Cassini Oval to saprolite directional ranges | Directional ranges embedded in the script |
| `src/nls_fitting_limonite_gaussian.py` | Gaussian | NLS fitting of Ellipse and Cassini Oval to Gaussian limonite directional ranges | Directional ranges embedded in the script |
| `src/nls_fitting_saprolite_gaussian.py` | Gaussian | NLS fitting of Ellipse and Cassini Oval to Gaussian saprolite directional ranges | Directional ranges embedded in the script |
| `src/ordinary_kriging_limonite.py` | Raw | Ordinary Kriging and related diagnostics for limonite | `example/synthetic_thickness.xlsx` |
| `src/ordinary_kriging_saprolite.py` | Raw | Ordinary Kriging and related diagnostics for saprolite | `example/synthetic_thickness.xlsx` |
| `src/sgs_limonite.py` | Gaussian | Sequential Gaussian Simulation and ensemble diagnostics for limonite | `example/synthetic_thickness.xlsx` |
| `src/sgs_saprolite.py` | Gaussian | Sequential Gaussian Simulation and ensemble diagnostics for saprolite | `example/synthetic_thickness.xlsx` |

## Parameter naming

The manuscript and public output tables report Cassini Oval parameters as `a`, `b`, and focus-axis azimuth. The symbol `b` is used consistently for the Cassini Oval constant in the public code outputs.

## Data note

The files in `example/` are synthetic and contain no original exploration data. The original drillhole dataset is excluded because it is subject to confidentiality restrictions.
