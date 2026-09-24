# Example data and quick test

This directory contains synthetic input data and a small repository test.

## Files

- `synthetic_thickness.xlsx` — synthetic Excel input used by the public OK/SGS examples.
- `synthetic_thickness.csv` — the same synthetic data in CSV format.
- `quick_test.py` — verifies the example input schema and smoke-tests Ellipse and Cassini Oval fitting functions.

The synthetic dataset contains the columns `X`, `Y`, `Limonite Thickness`, and `Saprolite Thickness`.

From the repository root, run:

```bash
python example/quick_test.py
```

Expected behavior: the script terminates without an exception and prints `Quick test passed.` followed by basic fitted-parameter information.
