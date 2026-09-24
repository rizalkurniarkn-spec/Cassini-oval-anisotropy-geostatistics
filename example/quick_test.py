"""Quick repository test using only public/synthetic inputs.

Run from the repository root:
    python example/quick_test.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import nls_fitting_saprolite as sap_nls

# 1) Verify the synthetic input schema used by the kriging/SGS scripts.
data = pd.read_excel(ROOT / "example" / "synthetic_thickness.xlsx")
required = {"X", "Y", "Ketebalan LIM", "Ketebalan SAP"}
missing = required.difference(data.columns)
assert not missing, f"Missing required columns: {sorted(missing)}"
assert len(data) >= 20, "Synthetic dataset is unexpectedly small."

# 2) Run a fast ellipse NLS smoke test using the public directional ranges.
predict, params = sap_nls.fit_ellipse(sap_nls.AZIMUTH_DEG, sap_nls.OBSERVED_RANGE_M)
pred = predict(sap_nls.AZIMUTH_DEG)
assert np.all(np.isfinite(pred)) and np.all(pred > 0)
assert np.isfinite(params["Major_m"]) and np.isfinite(params["Minor_m"])

# 3) Evaluate the Cassini implicit equation at public test coordinates.
values = sap_nls.cassini_implicit(
    np.array([0.0, 0.5, 1.0]),
    np.array([0.0, 0.25, 0.5]),
    0.7,
    0.9,
)
assert np.all(np.isfinite(values))

print("Quick test passed.")
print(f"Synthetic rows: {len(data)}")
print(f"Ellipse major/minor: {params['Major_m']:.3f} / {params['Minor_m']:.3f} m")
