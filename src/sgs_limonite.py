

# COMPARISON OF ELLIPSE VS CASSINI OVAL
# ============================================================
#
# Input:
#   Thickness.xlsx

#   X/Y coordinate columns are detected automatically (X/Easting and Y/Northing).
#
# Konsep:



#        nugget = 0.06
#        partial sill = 0.94
#        total sill = 1.00
#        model = spherical
#   4. The compared component is the directional-range geometry:
#        - Ellipse
#        - Oval Cassini affine-scaled

#      innovations to support a paired comparison of both geometries.



#






#   NMIN = 6
#   NMAX = 12

#
# Dependensi Anaconda:
#   numpy, pandas, scipy, matplotlib, openpyxl
#
# Catatan metodologi:





#     covariance memerlukan jitter.
# ============================================================

import os
import glob
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.optimize import minimize_scalar
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")

# Repository root used for portable example input/output paths.
def resolve_repo_root():
    """Return repository root in both terminal scripts and Jupyter notebooks."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]

    # Jupyter/IPython does not define __file__. Search the current working
    # directory and its parents for the repository structure.
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src").is_dir() and (candidate / "example").is_dir():
            return candidate
    return cwd


REPO_ROOT = resolve_repo_root()



# ============================================================

# ============================================================

DATA_FILE = str(REPO_ROOT / "example" / "synthetic_thickness.xlsx")
SHEET_NAME = 0

X_COL = None
Y_COL = None
Z_COL = "Limonite Thickness"


GRID_RES = 12.5



DRILL_SPACING_OVERRIDE = 25.0


BOUNDARY_BUFFER_MULTIPLIER = 1.0


BOUNDARY_CELL = 25.0
BOUNDARY_CLOSE_ITERS = 1
FILL_INTERNAL_BOUNDARY_HOLES = True
MASK_OUTSIDE_DATA_SHAPE = True


NMIN = 6
NMAX = 12
ALLOW_FALLBACK_NEAREST = True
NORMALIZED_SEARCH_RADIUS = 1.25


VARIOGRAM_MODEL = "spherical"
NUGGET = 0.06
PARTIAL_SILL = 0.94
SILL = NUGGET + PARTIAL_SILL

# SGS
N_REALIZATIONS = 100
BASE_SEED = 260815



HARD_CANDIDATE_K = 48
GRID_CANDIDATE_K = 128



HARD_MATCH_TOL = max(1e-8, GRID_RES * 1e-7)

# Stabilisasi numerik covariance lokal
COV_JITTER_START = 1e-10
COV_JITTER_MAX = 1e-2
VAR_MIN = 1e-10
VAR_MAX = SILL

# Back-transform tails:


BACKTRANSFORM_TAIL = "clip"

# ============================================================

# ============================================================


OUTPUT_DIR = REPO_ROOT / "outputs" / "sgs_limonite"

DIR_REALIZATION_CSV = OUTPUT_DIR / "01_realization_csv"
DIR_INDIVIDUAL_MAPS = OUTPUT_DIR / "02_map_individual"
DIR_COMPARE_MAPS = OUTPUT_DIR / "03_comparison_maps"
DIR_SUMMARY = OUTPUT_DIR / "04_spatial_summary"
DIR_ENSEMBLE = OUTPUT_DIR / "05_ensemble_diagnostics"
DIR_TABLES = OUTPUT_DIR / "06_diagnostic_tables"
DIR_ARRAYS = OUTPUT_DIR / "07_array_npy"

SAVE_ALL_REALIZATIONS = True
SAVE_EACH_REALIZATION_CSV = True
SAVE_INDIVIDUAL_REALIZATION_MAPS = True   # 100 Ellipse + 100 Cassini
SAVE_PAIRED_REALIZATION_MAPS = True       
SAVE_POSTPROCESS_CSV = True
SAVE_FIGS = True



SKIP_EXISTING_EXPORTS = True



SAVE_REALIZATION_CSV_DURING_SIMULATION = True

# Jangan tampilkan ratusan jendela saat dijalankan of Anaconda/Jupyter.
SHOW_PLOTS = False
RUN_LOOCV = True


EXAMPLE_REALIZATION = 1


N_LEVELS_MAP = 30
POINT_SIZE_SINGLE = 24
POINT_SIZE_COMPARE = 18
COLOR_SCALE_SINGLE = "global"   
COLOR_SCALE_COMPARE = "global"  # paired comparison memakai skala warna sama
MAP_DPI = 300






RUN_ENSEMBLE_DIAGNOSTICS = True
HISTOGRAM_BINS = 55
CDF_BINS = 500
VARIOGRAM_N_LAGS = 18
# None = otomatis: min(0.5 diagonal area, 1.5 x range directional terbesar).
VARIOGRAM_MAX_LAG = None
VARIOGRAM_PAIR_SAMPLE = 120000
VARIOGRAM_LOCAL_ANCHORS = 1000
VARIOGRAM_LOCAL_K = 100
VARIOGRAM_GLOBAL_PAIR_FACTOR = 4


RUN_DIRECTIONAL_VARIOGRAM_VALIDATION = True
DIRECTIONAL_TOLERANCE_DEG = 11.25   
DIRECTIONAL_RANGE_FIT_MIN = max(GRID_RES * 1.5, 20.0)
DIRECTIONAL_RANGE_FIT_MAX_FACTOR = 2.0
DIRECTIONAL_MIN_PAIRS_PER_BIN = 20

MODELS = ["Ellipse", "Cassini"]
EPS = 1e-12


def prepare_output_folders():
    """Create the output-folder structure."""
    folders = [
        OUTPUT_DIR,
        DIR_REALIZATION_CSV,
        DIR_INDIVIDUAL_MAPS / "Ellipse",
        DIR_INDIVIDUAL_MAPS / "Cassini",
        DIR_COMPARE_MAPS,
        DIR_SUMMARY,
        DIR_ENSEMBLE,
        DIR_TABLES,
        DIR_ARRAYS,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def output_file(path_like):
    """Return an output Path and ensure that its parent directory exists."""
    p = Path(path_like)
    if not p.is_absolute():
        p = OUTPUT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def should_export(path_like):
    """Return False when the file already exists and skip-existing mode is active."""
    p = output_file(path_like)
    return not (SKIP_EXISTING_EXPORTS and p.exists())


cmap_smooth = LinearSegmentedColormap.from_list(
    "smooth_blue_red",
    [
        "#0b3c8c",
        "#2c7fb8",
        "#41b6c4",
        "#a1dab4",
        "#ffffbf",
        "#fdae61",
        "#d7191c",
    ],
    N=256,
)


# ============================================================

# ============================================================

# Ellipse polar NLS
ELLIPSE_MAJOR = 293.99632947
ELLIPSE_MINOR = 163.69907027
ELLIPSE_PHI_AZ = 74.63410107
ELLIPSE_RATIO = ELLIPSE_MAJOR / ELLIPSE_MINOR

# Oval Cassini affine-scaled NLS
CASSINI_A = 165.56486891
CASSINI_B = 228.01740108
_CASSINI_INTERNAL_SCALE_X = 1.01083953
_CASSINI_INTERNAL_SCALE_Y = 0.98927671
CASSINI_PHI_AZ = 73.97365979

# Directional-range fitting statistics from the corresponding NLS script.
DIRECTIONAL_FIT_RMSE = {
    "Ellipse": 7.74743814,
    "Cassini": 4.52894832,
}

DIRECTIONAL_FIT_R2 = {
    "Ellipse": 0.97240331,
    "Cassini": 0.99056948,
}





DIRECTIONAL_VALIDATION_TABLE = pd.DataFrame({
    "Direction": ["N0E", "N22.5E", "N45E", "N67.5E", "N90E", "N112.5E", "N135E", "N157.5E"],
    "Azimuth_deg": [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5],
    "Original_Range_m": [166.0, 200.0, 241.0, 282.0, 276.0, 227.0, 170.0, 154.0],
})

DIRECTIONAL_AZIMUTHS = DIRECTIONAL_VALIDATION_TABLE["Azimuth_deg"].to_numpy(dtype=float)
DIRECTIONAL_ORIGINAL_RANGES = DIRECTIONAL_VALIDATION_TABLE["Original_Range_m"].to_numpy(dtype=float)

# ============================================================
# 3. UTILITAS DATA
# ============================================================

def find_excel_file(file_name):
    if os.path.exists(file_name):
        return file_name

    candidates = glob.glob("*Thickness*.xlsx") + glob.glob("*Thickness*.xls")

    if len(candidates) > 0:
        print(f"File {file_name} was not found. Using: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Input workbook was not found. Letakkan file Excel di folder "
        "as the script, or change DATA_FILE."
    )


def detect_column(df, candidates):
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]

    for c in cols:
        c_low = str(c).strip().lower()
        for cand in candidates:
            if str(cand).strip().lower() in c_low:
                return c

    return None


def load_thickness_data():
    file_path = find_excel_file(DATA_FILE)
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME)

    x_col = X_COL or detect_column(
        df, ["x", "easting", "east", "koordinat x", "x coordinate"]
    )
    y_col = Y_COL or detect_column(
        df, ["y", "northing", "north", "koordinat y", "y coordinate"]
    )
    z_col = Z_COL if Z_COL in df.columns else detect_column(
        df, ["Limonite Thickness", "limonite thickness", "lim thickness", "thickness"]
    )

    if x_col is None or y_col is None or z_col is None:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError(
            "X, Y, or Limonite Thickness column could not be detected. "
            "Set X_COL, Y_COL, and Z_COL manually."
        )

    data = df[[x_col, y_col, z_col]].copy()
    data.columns = ["X", "Y", "Z"]
    data[["X", "Y"]] = data[["X", "Y"]].apply(pd.to_numeric, errors="coerce")
    data["Z"] = pd.to_numeric(data["Z"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["X", "Y"]).reset_index(drop=True)

    if len(data) < NMIN + 1:
        raise ValueError(
            f"Only {len(data)} valid data records were found. "
            f"More than NMIN={NMIN} records are recommended."
        )

    
    
    n_before = len(data)
    data = data.groupby(["X", "Y"], as_index=False)["Z"].mean()
    n_after = len(data)

    if n_after < n_before:
        print(
            f"Warning: {n_before - n_after} duplicate-coordinate records "
            "were merged using the mean Limonite Thickness."
        )

    print("\n" + "=" * 72)
    print("DATA LIM")
    print("=" * 72)
    print(f"File                = {file_path}")
    print(f"X column            = {x_col}")
    print(f"Y column            = {y_col}")
    print(f"Simulation column    = {z_col}")
    print(f"Number of hard data    = {len(data)}")
    print(f"Min Limonite Thickness   = {data['Z'].min():.4f}")
    print(f"Max Limonite Thickness   = {data['Z'].max():.4f}")
    print(f"Mean Limonite Thickness  = {data['Z'].mean():.4f}")
    print(f"SD Limonite Thickness   = {data['Z'].std(ddof=1):.4f}")

    return data


# ============================================================

# ============================================================

def normal_score_transform(z):
    """
    Transformasi empiris rank -> standard normal.
    Ties memperoleh average rank.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)

    ranks = rankdata(z, method="average")
    probs = (ranks - 0.5) / n
    probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
    y = norm.ppf(probs)

    
    order = np.argsort(z)
    z_sorted = z[order]
    p_sorted = (np.arange(n) + 0.5) / n
    y_sorted = norm.ppf(p_sorted)

    return y, y_sorted, z_sorted


def back_transform(y_values, y_table, z_table, tail_mode="clip"):
    """Back-transform normal scores to the original thickness units."""
    y_values = np.asarray(y_values, dtype=float)

    if tail_mode == "clip":
        return np.interp(
            y_values,
            y_table,
            z_table,
            left=z_table[0],
            right=z_table[-1],
        )

    raise ValueError("BACKTRANSFORM_TAIL must currently be set to 'clip'.")


# ============================================================

# ============================================================

def azimuth_to_unit_vector(az_deg):
    """
    Geological azimuth:
      0° = North
      90° = East
      clockwise
    """
    az = np.deg2rad(az_deg)
    return np.sin(az), np.cos(az)


def pair_azimuth_from_dxdy(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def phi_az_to_theta_math(phi_az_deg):
    return np.deg2rad((90.0 - phi_az_deg) % 360.0)


def ellipse_range_from_azimuth(az_deg):
    az_deg = np.asarray(az_deg, dtype=float)

    ux, uy = azimuth_to_unit_vector(az_deg)
    phi = np.deg2rad(ELLIPSE_PHI_AZ)

    ex_major = np.sin(phi)
    ey_major = np.cos(phi)

    phi_minor = phi + np.pi / 2.0
    ex_minor = np.sin(phi_minor)
    ey_minor = np.cos(phi_minor)

    u_major = ux * ex_major + uy * ey_major
    u_minor = ux * ex_minor + uy * ey_minor

    denominator = np.sqrt(
        (u_major / ELLIPSE_MAJOR) ** 2
        + (u_minor / ELLIPSE_MINOR) ** 2
    )

    return 1.0 / np.maximum(denominator, EPS)


def cassini_range_from_azimuth(az_deg):
    """
    Directional range of affine-scaled Cassinian oval.

    [X_n^2 + Y_n^2 + a^2]^2 - 4 a^2 X_n^2 - b^4 = 0
    """
    az_deg = np.asarray(az_deg, dtype=float)

    ux, uy = azimuth_to_unit_vector(az_deg)
    theta = phi_az_to_theta_math(CASSINI_PHI_AZ)

    ct = np.cos(theta)
    st = np.sin(theta)

    local_x = ux * ct + uy * st
    local_y = -ux * st + uy * ct

    A = local_x / _CASSINI_INTERNAL_SCALE_X
    B = local_y / _CASSINI_INTERNAL_SCALE_Y

    sdir = A**2 + B**2

    qa = sdir**2
    qb = CASSINI_A**2 * (2.0 * sdir - 4.0 * A**2)
    qc = CASSINI_A**4 - CASSINI_B**4

    disc = qb**2 - 4.0 * qa * qc
    disc = np.where((disc < 0.0) & (disc > -1e-8), 0.0, disc)

    directional_range = np.full_like(az_deg, np.nan, dtype=float)
    valid_disc = disc >= 0.0

    if not np.any(valid_disc):
        return directional_range

    sqrt_disc = np.sqrt(np.where(valid_disc, disc, np.nan))

    q1 = (-qb + sqrt_disc) / (2.0 * qa)
    q2 = (-qb - sqrt_disc) / (2.0 * qa)

    q_candidates = np.stack([q1, q2], axis=0)
    q_candidates = np.where(q_candidates > 0.0, q_candidates, np.nan)

    valid_any = np.any(np.isfinite(q_candidates), axis=0)
    q = np.full_like(az_deg, np.nan, dtype=float)

    if np.any(valid_any):
        q[valid_any] = np.nanmax(q_candidates[:, valid_any], axis=0)

    directional_range[valid_any] = np.sqrt(q[valid_any])
    return directional_range


def get_directional_range(az_deg, model_name):
    if model_name == "Ellipse":
        return ellipse_range_from_azimuth(az_deg)
    if model_name == "Cassini":
        return cassini_range_from_azimuth(az_deg)
    raise ValueError("Unknown model name.")


def normalized_distance_from_dxdy(dx, dy, model_name):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    h = np.sqrt(dx**2 + dy**2)
    az = pair_azimuth_from_dxdy(dx, dy)
    directional_range = get_directional_range(az, model_name)

    out = np.full(np.broadcast(dx, dy).shape, np.inf, dtype=float)
    valid = np.isfinite(directional_range) & (directional_range > EPS)
    out[valid] = h[valid] / directional_range[valid]
    return out


# ============================================================

# ============================================================

def spherical_core(t):
    t = np.asarray(t, dtype=float)
    return np.where(
        t <= 1.0,
        1.5 * t - 0.5 * t**3,
        1.0,
    )


def semivariogram_from_dxdy(dx, dy, model_name):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    h = np.sqrt(dx**2 + dy**2)
    t = normalized_distance_from_dxdy(dx, dy, model_name)

    gamma = np.where(
        h <= EPS,
        0.0,
        NUGGET + PARTIAL_SILL * spherical_core(t),
    )

    return np.minimum(gamma, SILL)


def covariance_from_dxdy(dx, dy, model_name):
    """
    C(h) = sill - gamma(h), with C(0) = sill.
    """
    gamma = semivariogram_from_dxdy(dx, dy, model_name)
    cov = SILL - gamma
    return np.maximum(cov, 0.0)


# ============================================================

# ============================================================

class OrthogonalBoundary:
    """Construct an orthogonal boundary without Shapely."""
    def __init__(self, x_edges, y_edges, mask):
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.bounds = (
            self.x_edges[0], self.y_edges[0],
            self.x_edges[-1], self.y_edges[-1],
        )
        dx = np.diff(self.x_edges)
        dy = np.diff(self.y_edges)
        self.area = float(np.sum(self.mask * dy[:, None] * dx[None, :]))
        self.segments = self._segments()

    def _segments(self):
        segments = []
        m = self.mask
        xe, ye = self.x_edges, self.y_edges
        ny, nx = m.shape
        for i, j in np.argwhere(m):
            if j == 0 or not m[i, j - 1]:
                segments.append(((xe[j], ye[i]), (xe[j], ye[i + 1])))
            if j == nx - 1 or not m[i, j + 1]:
                segments.append(((xe[j + 1], ye[i]), (xe[j + 1], ye[i + 1])))
            if i == 0 or not m[i - 1, j]:
                segments.append(((xe[j], ye[i]), (xe[j + 1], ye[i])))
            if i == ny - 1 or not m[i + 1, j]:
                segments.append(((xe[j], ye[i + 1]), (xe[j + 1], ye[i + 1])))
        return segments


def estimate_drill_spacing(coords):
    """Estimate typical drill spacing from the median nearest-neighbor distance."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return float(GRID_RES)
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    nn = np.asarray(dist[:, 1], dtype=float)
    nn = nn[np.isfinite(nn) & (nn > EPS)]
    return float(np.median(nn)) if len(nn) else float(GRID_RES)


def build_orthogonal_boundary(coords, offset, cell=BOUNDARY_CELL):
    """
    Boundary formed from the union of boxes with +/- offset around drillholes.
    The boundary raster uses a regular 25 m cell size followed by closing and hole filling
    to obtain a clean horizontal/vertical boundary without internal holes.
    """
    coords = np.asarray(coords, dtype=float)
    minx = np.floor((coords[:, 0].min() - offset) / cell) * cell
    maxx = np.ceil((coords[:, 0].max() + offset) / cell) * cell
    miny = np.floor((coords[:, 1].min() - offset) / cell) * cell
    maxy = np.ceil((coords[:, 1].max() + offset) / cell) * cell

    x_edges = np.arange(minx, maxx + cell, cell, dtype=float)
    y_edges = np.arange(miny, maxy + cell, cell, dtype=float)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    mask = np.zeros((ny, nx), dtype=bool)

    for x, y in coords:
        ix0 = max(0, np.searchsorted(x_edges, x - offset, side="right") - 1)
        ix1 = min(nx, np.searchsorted(x_edges, x + offset, side="left"))
        iy0 = max(0, np.searchsorted(y_edges, y - offset, side="right") - 1)
        iy1 = min(ny, np.searchsorted(y_edges, y + offset, side="left"))
        mask[iy0:iy1, ix0:ix1] = True

    
    if BOUNDARY_CLOSE_ITERS > 0:
        p = BOUNDARY_CLOSE_ITERS + 1
        padded = np.pad(mask, p, mode="constant", constant_values=False)
        padded = binary_closing(
            padded,
            structure=np.ones((3, 3), dtype=bool),
            iterations=BOUNDARY_CLOSE_ITERS,
        )
        mask = padded[p:-p, p:-p]

    if FILL_INTERNAL_BOUNDARY_HOLES:
        mask = binary_fill_holes(mask)

    return OrthogonalBoundary(x_edges, y_edges, mask)


def points_inside_boundary(boundary, xy):
    """Test points against boundary cells; points exactly on an edge are retained."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    ix_a = np.searchsorted(boundary.x_edges, x, side="right") - 1
    ix_b = np.searchsorted(boundary.x_edges, x, side="left") - 1
    iy_a = np.searchsorted(boundary.y_edges, y, side="right") - 1
    iy_b = np.searchsorted(boundary.y_edges, y, side="left") - 1

    inside = np.zeros(len(xy), dtype=bool)
    ny, nx = boundary.mask.shape
    for ix in (ix_a, ix_b):
        for iy in (iy_a, iy_b):
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            if np.any(ok):
                inside[ok] |= boundary.mask[iy[ok], ix[ok]]
    return inside


def build_grid(data):
    """Create a 12.5 m SGS grid inside the 25 m orthogonal boundary without internal holes."""
    coords = data[["X", "Y"]].to_numpy(dtype=float)
    spacing_auto = estimate_drill_spacing(coords)
    drill_spacing = (
        float(DRILL_SPACING_OVERRIDE)
        if DRILL_SPACING_OVERRIDE is not None
        else spacing_auto
    )
    boundary_buffer = float(BOUNDARY_BUFFER_MULTIPLIER) * drill_spacing

    boundary = build_orthogonal_boundary(
        coords,
        offset=boundary_buffer,
        cell=BOUNDARY_CELL,
    )
    minx, miny, maxx, maxy = boundary.bounds

    xmin = np.floor(minx / GRID_RES) * GRID_RES
    xmax = np.ceil(maxx / GRID_RES) * GRID_RES
    ymin = np.floor(miny / GRID_RES) * GRID_RES
    ymax = np.ceil(maxy / GRID_RES) * GRID_RES

    x_grid = np.arange(xmin, xmax + GRID_RES * 0.5, GRID_RES)
    y_grid = np.arange(ymin, ymax + GRID_RES * 0.5, GRID_RES)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])

    if MASK_OUTSIDE_DATA_SHAPE:
        inside_mask = points_inside_boundary(boundary, grid_points)
    else:
        inside_mask = np.ones(len(grid_points), dtype=bool)

    valid_flat_idx = np.where(inside_mask)[0]
    valid_points = grid_points[valid_flat_idx]
    if len(valid_points) == 0:
        raise ValueError("The boundary produced zero simulation nodes.")

    print("\n" + "=" * 72)
    print("GRID SGS - BOUNDARY ORTOGONAL RAPI")
    print("=" * 72)
    print(f"GRID_RES                  = {GRID_RES:.3f} m")
    print(f"Automatic drill spacing (median NN)= {spacing_auto:.3f} m")
    print(f"Drill spacing used         = {drill_spacing:.3f} m")
    print(f"Buffer boundary           = {boundary_buffer:.3f} m")
    print(f"Boundary cell             = {BOUNDARY_CELL:.3f} m")
    print(f"Boundary closing iter     = {BOUNDARY_CLOSE_ITERS}")
    print(f"Fill internal holes       = {FILL_INTERNAL_BOUNDARY_HOLES}")
    print(f"Luas boundary             = {boundary.area:.2f} m2")
    print(f"Number of rectangular grid nodes     = {len(grid_points)}")
    print(f"Simulated grid nodes        = {len(valid_points)}")

    return {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "XX": XX,
        "YY": YY,
        "grid_points": grid_points,
        "inside_mask": inside_mask,
        "core_mask": inside_mask.copy(),
        "valid_flat_idx": valid_flat_idx,
        "valid_points": valid_points,
        "drill_spacing_auto": spacing_auto,
        "drill_spacing_used": drill_spacing,
        "boundary_buffer": boundary_buffer,
        "boundary": boundary,
    }


def plot_orthogonal_boundary(ax, grid_info, label="Boundary SGS (+/-25 m)", linewidth=1.6):
    """Add the orthogonal boundary outline to a map."""
    boundary = grid_info.get("boundary")
    if boundary is None or not boundary.segments:
        return
    lc = LineCollection(
        boundary.segments,
        colors="black",
        linewidths=linewidth,
        zorder=10,
        label=label,
    )
    ax.add_collection(lc)


# ============================================================

# ============================================================

def ensure_2d_query_result(values, n_rows):
    arr = np.asarray(values)
    if arr.ndim == 1:
        if n_rows == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(n_rows, 1)
    return arr


def build_candidate_cache(valid_points, hard_coords, model_name):
    """
    Speed up SGS by precomputing candidate hard-data and grid neighbors.
    Candidate ordering is determined by normalized anisotropic distance.
    """
    n_grid = len(valid_points)
    n_hard = len(hard_coords)

    print(f"\nPrecompute candidate cache: {model_name}")

    # ---------------- HARD DATA ----------------
    hard_tree = cKDTree(hard_coords)
    k_hard = min(max(NMAX, HARD_CANDIDATE_K), n_hard)

    hard_dist_e, hard_idx = hard_tree.query(valid_points, k=k_hard)
    hard_idx = ensure_2d_query_result(hard_idx, n_grid).astype(np.int32)

    hard_candidate_coords = hard_coords[hard_idx]
    dx_h = hard_candidate_coords[:, :, 0] - valid_points[:, None, 0]
    dy_h = hard_candidate_coords[:, :, 1] - valid_points[:, None, 1]
    hard_dnorm = normalized_distance_from_dxdy(dx_h, dy_h, model_name)

    hard_order = np.argsort(hard_dnorm, axis=1)
    hard_idx = np.take_along_axis(hard_idx, hard_order, axis=1)
    hard_dnorm = np.take_along_axis(hard_dnorm, hard_order, axis=1)

    
    grid_tree = cKDTree(valid_points)

    if n_grid > 1:
        k_grid = min(GRID_CANDIDATE_K + 1, n_grid)
        grid_dist_e, grid_idx = grid_tree.query(valid_points, k=k_grid)
        grid_idx = ensure_2d_query_result(grid_idx, n_grid).astype(np.int32)

        grid_candidate_coords = valid_points[grid_idx]
        dx_g = grid_candidate_coords[:, :, 0] - valid_points[:, None, 0]
        dy_g = grid_candidate_coords[:, :, 1] - valid_points[:, None, 1]
        grid_dnorm = normalized_distance_from_dxdy(dx_g, dy_g, model_name)

        # Buang node target sendiri.
        self_mask = grid_idx == np.arange(n_grid, dtype=np.int32)[:, None]
        grid_dnorm[self_mask] = np.inf

        grid_order = np.argsort(grid_dnorm, axis=1)
        grid_idx = np.take_along_axis(grid_idx, grid_order, axis=1)
        grid_dnorm = np.take_along_axis(grid_dnorm, grid_order, axis=1)

        
        keep = min(GRID_CANDIDATE_K, grid_idx.shape[1])
        grid_idx = grid_idx[:, :keep]
        grid_dnorm = grid_dnorm[:, :keep]
    else:
        grid_idx = np.empty((1, 0), dtype=np.int32)
        grid_dnorm = np.empty((1, 0), dtype=float)

    
    exact_dist, exact_idx = hard_tree.query(valid_points, k=1)

    print(f"  hard candidate K    = {hard_idx.shape[1]}")
    print(f"  grid candidate K    = {grid_idx.shape[1]}")

    return {
        "hard_idx": hard_idx,
        "hard_dnorm": hard_dnorm.astype(np.float32),
        "grid_idx": grid_idx,
        "grid_dnorm": grid_dnorm.astype(np.float32),
        "exact_hard_dist": np.asarray(exact_dist, dtype=float),
        "exact_hard_idx": np.asarray(exact_idx, dtype=np.int32),
    }


# ============================================================
# 9. PEMILIHAN NEIGHBOR DINAMIS SGS
# ============================================================

def select_sgs_neighbors(
    local_grid_index,
    valid_points,
    hard_coords,
    hard_gauss,
    sim_values,
    simulated_flag,
    cache,
):
    """
    Menggabungkan:
      - hard data;
      - grid nodes already simulated along the random path.

    Priority follows normalized anisotropic distance.
    """
    h_idx = cache["hard_idx"][local_grid_index]
    h_dn = cache["hard_dnorm"][local_grid_index].astype(float)

    g_idx_all = cache["grid_idx"][local_grid_index]
    g_dn_all = cache["grid_dnorm"][local_grid_index].astype(float)

    if len(g_idx_all) > 0:
        g_available = (
            np.isfinite(g_dn_all)
            & (g_idx_all >= 0)
            & simulated_flag[g_idx_all]
        )
        g_idx = g_idx_all[g_available]
        g_dn = g_dn_all[g_available]
    else:
        g_idx = np.empty(0, dtype=np.int32)
        g_dn = np.empty(0, dtype=float)

    
    cand_type = np.concatenate([
        np.zeros(len(h_idx), dtype=np.int8),
        np.ones(len(g_idx), dtype=np.int8),
    ])

    cand_index = np.concatenate([
        h_idx.astype(np.int32),
        g_idx.astype(np.int32),
    ])

    cand_dn = np.concatenate([h_dn, g_dn])

    finite = np.isfinite(cand_dn)
    cand_type = cand_type[finite]
    cand_index = cand_index[finite]
    cand_dn = cand_dn[finite]

    if len(cand_dn) == 0:
        return None

    inside = cand_dn <= NORMALIZED_SEARCH_RADIUS

    if np.sum(inside) >= NMIN:
        available_mask = inside
    elif ALLOW_FALLBACK_NEAREST:
        available_mask = np.ones(len(cand_dn), dtype=bool)
    else:
        return None

    idx_available = np.where(available_mask)[0]
    order = idx_available[np.argsort(cand_dn[idx_available])]
    chosen = order[: min(NMAX, len(order))]

    if len(chosen) == 0:
        return None

    types = cand_type[chosen]
    indices = cand_index[chosen]

    neighbor_coords = np.empty((len(chosen), 2), dtype=float)
    neighbor_values = np.empty(len(chosen), dtype=float)

    hard_mask = types == 0
    sim_mask = types == 1

    if np.any(hard_mask):
        hi = indices[hard_mask]
        neighbor_coords[hard_mask] = hard_coords[hi]
        neighbor_values[hard_mask] = hard_gauss[hi]

    if np.any(sim_mask):
        gi = indices[sim_mask]
        neighbor_coords[sim_mask] = valid_points[gi]
        neighbor_values[sim_mask] = sim_values[gi]

    return neighbor_coords, neighbor_values


# ============================================================

# ============================================================

def simple_kriging_gaussian(target_xy, neighbor_coords, neighbor_values, model_name):
    """
    Simple Kriging di ruang normal-score:
      mean global Gaussian = 0
      variance global = SILL = 1
    """
    n = len(neighbor_values)

    if n == 0:
        return 0.0, SILL, 0.0, False

    dx_mat = neighbor_coords[None, :, 0] - neighbor_coords[:, None, 0]
    dy_mat = neighbor_coords[None, :, 1] - neighbor_coords[:, None, 1]

    C = covariance_from_dxdy(dx_mat, dy_mat, model_name)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, SILL)

    dx0 = neighbor_coords[:, 0] - target_xy[0]
    dy0 = neighbor_coords[:, 1] - target_xy[1]
    c0 = covariance_from_dxdy(dx0, dy0, model_name)

    jitter = 0.0
    repaired = False
    solved = False

    # Attempt a direct solve, then add numerical jitter progressively if needed.
    trial_jitters = [0.0]
    j = COV_JITTER_START
    while j <= COV_JITTER_MAX:
        trial_jitters.append(j)
        j *= 10.0

    weights = None
    cond_var = np.nan

    for jitter_try in trial_jitters:
        C_try = C.copy()
        if jitter_try > 0.0:
            C_try.flat[:: n + 1] += jitter_try

        try:
            weights_try = np.linalg.solve(C_try, c0)
        except np.linalg.LinAlgError:
            continue

        cond_var_try = SILL - float(np.dot(weights_try, c0))

        
        if np.isfinite(cond_var_try) and cond_var_try >= -1e-8:
            weights = weights_try
            cond_var = cond_var_try
            jitter = jitter_try
            repaired = jitter_try > 0.0
            solved = True
            break

    if not solved:
        # Fallback least squares + variance clip.
        C_try = C.copy()
        C_try.flat[:: n + 1] += COV_JITTER_MAX
        weights = np.linalg.lstsq(C_try, c0, rcond=None)[0]
        cond_var = SILL - float(np.dot(weights, c0))
        jitter = COV_JITTER_MAX
        repaired = True

    cond_mean = float(np.dot(weights, neighbor_values))

    variance_clipped = False
    if not np.isfinite(cond_var):
        cond_var = SILL
        variance_clipped = True
    elif cond_var < VAR_MIN:
        cond_var = VAR_MIN
        variance_clipped = True
    elif cond_var > VAR_MAX:
        cond_var = VAR_MAX
        variance_clipped = True

    return cond_mean, cond_var, jitter, (repaired or variance_clipped)


# ============================================================

# ============================================================

def simulate_one_realization(
    model_name,
    valid_points,
    hard_coords,
    hard_gauss,
    cache,
    random_path,
    gaussian_innovations,
):
    n_grid = len(valid_points)

    sim_values = np.full(n_grid, np.nan, dtype=float)
    simulated_flag = np.zeros(n_grid, dtype=bool)

    n_repaired = 0
    max_jitter = 0.0
    n_exact_hard = 0

    for step, local_idx in enumerate(random_path):
        
        if cache["exact_hard_dist"][local_idx] <= HARD_MATCH_TOL:
            hidx = cache["exact_hard_idx"][local_idx]
            sim_values[local_idx] = hard_gauss[hidx]
            simulated_flag[local_idx] = True
            n_exact_hard += 1
            continue

        neighbors = select_sgs_neighbors(
            local_idx,
            valid_points,
            hard_coords,
            hard_gauss,
            sim_values,
            simulated_flag,
            cache,
        )

        if neighbors is None:
            cond_mean = 0.0
            cond_var = SILL
            jitter = 0.0
            repaired = False
        else:
            neighbor_coords, neighbor_values = neighbors
            cond_mean, cond_var, jitter, repaired = simple_kriging_gaussian(
                valid_points[local_idx],
                neighbor_coords,
                neighbor_values,
                model_name,
            )

        sim_values[local_idx] = (
            cond_mean + np.sqrt(max(cond_var, VAR_MIN)) * gaussian_innovations[step]
        )
        simulated_flag[local_idx] = True

        if repaired:
            n_repaired += 1
        if jitter > max_jitter:
            max_jitter = jitter

    diagnostics = {
        "n_repaired": n_repaired,
        "max_jitter": max_jitter,
        "n_exact_hard": n_exact_hard,
    }

    return sim_values, diagnostics


# ============================================================

# ============================================================

def loocv_model(hard_coords, hard_gauss, z_original, y_table, z_table, model_name):
    pred_gauss = np.full(len(hard_gauss), np.nan, dtype=float)
    pred_z = np.full(len(hard_gauss), np.nan, dtype=float)

    for i in range(len(hard_gauss)):
        train_mask = np.ones(len(hard_gauss), dtype=bool)
        train_mask[i] = False

        train_coords = hard_coords[train_mask]
        train_values = hard_gauss[train_mask]

        dx = train_coords[:, 0] - hard_coords[i, 0]
        dy = train_coords[:, 1] - hard_coords[i, 1]
        dnorm = normalized_distance_from_dxdy(dx, dy, model_name)

        finite = np.isfinite(dnorm)
        inside = finite & (dnorm <= NORMALIZED_SEARCH_RADIUS)

        if np.sum(inside) >= NMIN:
            candidates = np.where(inside)[0]
        else:
            candidates = np.where(finite)[0]

        if len(candidates) == 0:
            continue

        order = candidates[np.argsort(dnorm[candidates])]
        chosen = order[: min(NMAX, len(order))]

        mean_g, var_g, jitter, repaired = simple_kriging_gaussian(
            hard_coords[i],
            train_coords[chosen],
            train_values[chosen],
            model_name,
        )

        pred_gauss[i] = mean_g
        pred_z[i] = back_transform(
            np.array([mean_g]), y_table, z_table, BACKTRANSFORM_TAIL
        )[0]

    valid = np.isfinite(pred_z)
    err = pred_z[valid] - z_original[valid]

    rmse = np.sqrt(np.mean(err**2)) if np.any(valid) else np.nan
    mae = np.mean(np.abs(err)) if np.any(valid) else np.nan
    me = np.mean(err) if np.any(valid) else np.nan
    r2 = np.nan

    if np.sum(valid) > 1:
        ss_res = np.sum(err**2)
        ss_tot = np.sum((z_original[valid] - np.mean(z_original[valid])) ** 2)
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot

    detail = pd.DataFrame({
        "X": hard_coords[:, 0],
        "Y": hard_coords[:, 1],
        "Observed": z_original,
        "Predicted": pred_z,
        "Error": pred_z - z_original,
    })

    summary = {
        "Model": model_name,
        "CV_RMSE": rmse,
        "CV_MAE": mae,
        "CV_ME": me,
        "CV_R2": r2,
        "N_CV": int(np.sum(valid)),
    }

    return summary, detail


# ============================================================
# 13. POST-PROCESSING
# ============================================================

def postprocess_realizations(realizations):
    """realizations shape = (n_real, n_valid_grid)."""
    etype = np.nanmean(realizations, axis=0)
    sd = np.nanstd(realizations, axis=0, ddof=1)
    variance = sd ** 2
    p025 = np.nanpercentile(realizations, 2.5, axis=0)
    p25 = np.nanpercentile(realizations, 25, axis=0)
    p50 = np.nanpercentile(realizations, 50, axis=0)
    p75 = np.nanpercentile(realizations, 75, axis=0)
    p95 = np.nanpercentile(realizations, 95, axis=0)
    p975 = np.nanpercentile(realizations, 97.5, axis=0)
    cl95_width = p975 - p025

    return {
        "Etype": etype,
        "SD": sd,
        "Variance": variance,
        "CL95_Lower": p025,
        "CL95_Upper": p975,
        "CL95_Width": cl95_width,
        "P25": p25,
        "P50": p50,
        "P75": p75,
        "P95": p95,
    }


def valid_to_full_grid(values_valid, grid_info):
    full = np.full(len(grid_info["grid_points"]), np.nan, dtype=float)
    full[grid_info["valid_flat_idx"]] = values_valid
    return full.reshape(grid_info["XX"].shape)


# ============================================================
# 14. PLOT
# ============================================================

def _safe_levels(vmin, vmax, n_levels=N_LEVELS_MAP):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return n_levels
    if abs(vmax - vmin) <= EPS:
        delta = max(abs(vmin) * 0.01, 0.01)
        return np.linspace(vmin - delta, vmax + delta, n_levels)
    return np.linspace(vmin, vmax, n_levels)


def _add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.08, 0.94),
        xytext=(0.08, 0.81),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.5),
    )


def plot_single_sgs_map(
    Zmap,
    grid_info,
    hard_coords,
    z_original,
    model_name,
    realization_number,
    file_name,
    global_vmin=None,
    global_vmax=None,
):
    """Save one SGS map using the same style as the estimation script."""
    XX = grid_info["XX"]
    YY = grid_info["YY"]

    finite = Zmap[np.isfinite(Zmap)]
    if len(finite) == 0:
        return

    if COLOR_SCALE_SINGLE == "global" and global_vmin is not None:
        vmin_map = global_vmin
        vmax_map = global_vmax
    else:
        vmin_map = float(np.nanmin(finite))
        vmax_map = float(np.nanmax(finite))

    levels = _safe_levels(vmin_map, vmax_map)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    cf = ax.contourf(
        XX,
        YY,
        Zmap,
        levels=levels,
        cmap=cmap_smooth,
        vmin=vmin_map,
        vmax=vmax_map,
        extend="both",
    )

    ax.scatter(
        hard_coords[:, 0],
        hard_coords[:, 1],
        c=z_original,
        cmap=cmap_smooth,
        edgecolor="black",
        linewidth=0.4,
        s=POINT_SIZE_SINGLE,
        vmin=vmin_map,
        vmax=vmax_map,
        label="Drillhole data",
        zorder=5,
    )

    plot_orthogonal_boundary(ax, grid_info)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("SGS Limonite Thickness")

    ax.set_title(
        f"{model_name} - SGS Limonite Thickness - Realization {realization_number:03d}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", fontsize=8)
    _add_north_arrow(ax)

    fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison_sgs_map(
    maps_by_model,
    grid_info,
    hard_coords,
    z_original,
    realization_number,
    file_name,
    global_vmin,
    global_vmax,
):
    """Create a paired-realization map for both models using the same color scale."""
    XX = grid_info["XX"]
    YY = grid_info["YY"]

    if COLOR_SCALE_COMPARE == "global":
        vmin_compare = global_vmin
        vmax_compare = global_vmax
    else:
        vals = np.concatenate([
            maps_by_model[m][np.isfinite(maps_by_model[m])]
            for m in MODELS
        ])
        vmin_compare = float(np.nanmin(vals))
        vmax_compare = float(np.nanmax(vals))

    levels = _safe_levels(vmin_compare, vmax_compare)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)

    cf = None
    for ax, model_name in zip(axes, MODELS):
        Zmap = maps_by_model[model_name]
        cf = ax.contourf(
            XX, YY, Zmap,
            levels=levels,
            cmap=cmap_smooth,
            vmin=vmin_compare,
            vmax=vmax_compare,
            extend="both",
        )
        ax.scatter(
            hard_coords[:, 0], hard_coords[:, 1],
            c=z_original,
            cmap=cmap_smooth,
            edgecolor="black",
            linewidth=0.35,
            s=POINT_SIZE_COMPARE,
            vmin=vmin_compare,
            vmax=vmax_compare,
            zorder=5,
        )
        plot_orthogonal_boundary(ax, grid_info, label=None, linewidth=1.25)
        ax.set_title(
            f"Model {model_name} - Realization {realization_number:03d}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        _add_north_arrow(ax)

    if cf is not None:
        fig.colorbar(
            cf,
            ax=axes.ravel().tolist(),
            label="SGS Limonite Thickness",
            shrink=0.88,
        )

    fig.suptitle(
        f"SGS Comparison Limonite Thickness - Realization {realization_number:03d}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _plot_drill_locations(ax, hard_coords, label="Drillhole data"):
    """Use consistent drillhole-location markers on all maps."""
    ax.scatter(
        hard_coords[:, 0],
        hard_coords[:, 1],
        s=18,
        facecolors="white",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.95,
        label=label,
        zorder=8,
    )


def plot_model_maps(
    maps_by_model,
    grid_info,
    hard_coords,
    title_by_model,
    suptitle,
    file_name,
    colorbar_label="Limonite Thickness",
    fixed_vmin=None,
    fixed_vmax=None,
):
    """Compare three maps with drillhole markers on every panel.

    When fixed_vmin/fixed_vmax are provided, all panels use the same color range
    for direct comparison in thickness units.
    """
    XX = grid_info["XX"]
    YY = grid_info["YY"]

    finite_chunks = [
        maps_by_model[m][np.isfinite(maps_by_model[m])]
        for m in MODELS
        if np.any(np.isfinite(maps_by_model[m]))
    ]
    if not finite_chunks:
        return

    finite_values = np.concatenate(finite_chunks)
    vmin = float(fixed_vmin) if fixed_vmin is not None else float(np.nanmin(finite_values))
    vmax = float(fixed_vmax) if fixed_vmax is not None else float(np.nanmax(finite_values))
    levels = _safe_levels(vmin, vmax)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    cf = None

    for ax, model_name in zip(axes, MODELS):
        Zmap = maps_by_model[model_name]
        cf = ax.contourf(
            XX, YY, Zmap,
            levels=levels,
            cmap=cmap_smooth,
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )
        _plot_drill_locations(ax, hard_coords)
        plot_orthogonal_boundary(ax, grid_info, label=None, linewidth=1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title_by_model[model_name], fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=7)
        _add_north_arrow(ax)

    if cf is not None:
        fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.86, label=colorbar_label)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    if SAVE_FIGS:
        fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def plot_etype_differences(etype_maps, grid_info, hard_coords, file_name):
    """Difference map E-type Cassini - Ellipse."""
    XX = grid_info["XX"]
    YY = grid_info["YY"]
    diff_map = etype_maps["Cassini"] - etype_maps["Ellipse"]
    finite = diff_map[np.isfinite(diff_map)]
    if len(finite) == 0:
        return

    vmax_abs = float(np.nanmax(np.abs(finite)))
    if vmax_abs <= EPS:
        vmax_abs = 1.0
    levels = np.linspace(-vmax_abs, vmax_abs, 31)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cf = ax.contourf(XX, YY, diff_map, levels=levels, cmap="coolwarm", extend="both")
    _plot_drill_locations(ax, hard_coords)
    plot_orthogonal_boundary(ax, grid_info, label=None, linewidth=1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cassini - Ellipse", fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=7)
    _add_north_arrow(ax)
    fig.colorbar(cf, ax=ax, shrink=0.86, label="Difference E-type")
    fig.suptitle("Difference E-type SGS LIM: Cassini - Ellipse", fontsize=13, fontweight="bold")

    if SAVE_FIGS:
        fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def build_variogram_pair_sample(valid_points):
    """Sample common grid pairs for ensemble variograms.

    Pairs combine local and global samples. Lag distance uses Euclidean distance
    while direction uses axial geological azimuth from 0 to 180 degrees.
    """
    points = np.asarray(valid_points, dtype=float)
    n = len(points)
    if n < 2:
        raise ValueError("At least two grid nodes are required for the ensemble variogram.")

    rng = np.random.default_rng(BASE_SEED + 99173)
    pair_blocks = []

    
    tree = cKDTree(points)
    n_anchor = min(VARIOGRAM_LOCAL_ANCHORS, n)
    anchors = rng.choice(n, size=n_anchor, replace=False)
    k = min(max(2, VARIOGRAM_LOCAL_K), n)
    _, idx = tree.query(points[anchors], k=k)
    idx = ensure_2d_query_result(idx, n_anchor).astype(np.int64)
    a = np.repeat(anchors, idx.shape[1])
    b = idx.ravel()
    mask = a != b
    pair_blocks.append(np.column_stack([a[mask], b[mask]]))

    
    n_global = max(
        VARIOGRAM_PAIR_SAMPLE,
        VARIOGRAM_PAIR_SAMPLE * VARIOGRAM_GLOBAL_PAIR_FACTOR,
    )
    a = rng.integers(0, n, size=n_global, dtype=np.int64)
    b = rng.integers(0, n, size=n_global, dtype=np.int64)
    mask = a != b
    pair_blocks.append(np.column_stack([a[mask], b[mask]]))

    pairs = np.vstack(pair_blocks)
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)

    dx = points[pairs[:, 1], 0] - points[pairs[:, 0], 0]
    dy = points[pairs[:, 1], 1] - points[pairs[:, 0], 1]
    h = np.sqrt(dx**2 + dy**2)
    az = pair_azimuth_from_dxdy(dx, dy) % 180.0

    diagonal = float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
    if VARIOGRAM_MAX_LAG is None:
        max_target_range = float(np.max(DIRECTIONAL_ORIGINAL_RANGES))
        max_lag = min(0.5 * diagonal, 1.5 * max_target_range)
    else:
        max_lag = float(VARIOGRAM_MAX_LAG)

    valid = np.isfinite(h) & (h > EPS) & (h <= max_lag)
    pairs = pairs[valid]
    h = h[valid]
    dx = dx[valid]
    dy = dy[valid]
    az = az[valid]

    edges = np.linspace(0.0, max_lag, VARIOGRAM_N_LAGS + 1)
    bin_id = np.digitize(h, edges, right=False) - 1
    valid_bin = (bin_id >= 0) & (bin_id < VARIOGRAM_N_LAGS)
    pairs = pairs[valid_bin]
    h = h[valid_bin]
    dx = dx[valid_bin]
    dy = dy[valid_bin]
    az = az[valid_bin]
    bin_id = bin_id[valid_bin]

    # Stratified sampling per lag bin.
    quota = max(1, int(np.ceil(VARIOGRAM_PAIR_SAMPLE / VARIOGRAM_N_LAGS)))
    chosen_parts = []
    for b in range(VARIOGRAM_N_LAGS):
        ids = np.where(bin_id == b)[0]
        if len(ids) > quota:
            ids = rng.choice(ids, size=quota, replace=False)
        chosen_parts.append(ids)
    chosen = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=int)

    pairs = pairs[chosen]
    h = h[chosen]
    dx = dx[chosen]
    dy = dy[chosen]
    az = az[chosen]
    bin_id = bin_id[chosen]

    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.bincount(bin_id, minlength=VARIOGRAM_N_LAGS)

    print("\nVARIOGRAM ENSEMBLE PAIR SAMPLE")
    print(f"Number of pairs used = {len(pairs)}")
    print(f"Lag maksimum            = {max_lag:.3f} m")
    print(f"Number of lag bins          = {VARIOGRAM_N_LAGS}")

    return {
        "pairs": pairs.astype(np.int32),
        "h": h,
        "dx": dx,
        "dy": dy,
        "azimuth_axial": az,
        "bin_id": bin_id.astype(np.int16),
        "edges": edges,
        "centers": centers,
        "counts": counts,
        "max_lag": max_lag,
    }

def experimental_variogram_from_pairs(values_gauss, pair_info):
    """Compute the experimental variogram of one realization on common lag bins."""
    values = np.asarray(values_gauss, dtype=float)
    pairs = pair_info["pairs"]
    bins = pair_info["bin_id"].astype(int)
    diff = values[pairs[:, 0]] - values[pairs[:, 1]]
    semi = 0.5 * diff**2
    sums = np.bincount(bins, weights=semi, minlength=VARIOGRAM_N_LAGS)
    counts = np.bincount(bins, minlength=VARIOGRAM_N_LAGS)
    out = np.full(VARIOGRAM_N_LAGS, np.nan, dtype=float)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out


def theoretical_omnidirectional_variogram(lag_centers, model_name):
    """Return the angularly averaged theoretical variogram at physical lag distances."""
    az = np.linspace(0.0, 180.0, 181, endpoint=False)
    az_rad = np.deg2rad(az)
    out = np.empty(len(lag_centers), dtype=float)
    for i, h in enumerate(lag_centers):
        dx = h * np.sin(az_rad)
        dy = h * np.cos(az_rad)
        out[i] = float(np.mean(semivariogram_from_dxdy(dx, dy, model_name)))
    return out



def axial_angular_difference(azimuth_deg, target_deg):
    """Axial angular difference on 0-180 degrees; directions d and d+180 are equivalent."""
    azimuth_deg = np.asarray(azimuth_deg, dtype=float) % 180.0
    target_deg = float(target_deg) % 180.0
    return np.abs(((azimuth_deg - target_deg + 90.0) % 180.0) - 90.0)


def build_directional_pair_infos(pair_info):
    """Derive eight directional pair subsets from the common grid-pair sample."""
    infos = []
    az_all = np.asarray(pair_info["azimuth_axial"], dtype=float)
    bins_all = np.asarray(pair_info["bin_id"], dtype=int)

    for i, row in DIRECTIONAL_VALIDATION_TABLE.iterrows():
        az0 = float(row["Azimuth_deg"])
        angle_diff = axial_angular_difference(az_all, az0)
        mask = angle_diff <= (DIRECTIONAL_TOLERANCE_DEG + 1e-12)

        pairs = pair_info["pairs"][mask]
        bins = bins_all[mask]
        h = pair_info["h"][mask]
        counts = np.bincount(bins, minlength=VARIOGRAM_N_LAGS)

        infos.append({
            "direction_index": int(i),
            "direction_name": str(row["Direction"]),
            "azimuth_deg": az0,
            "original_range_m": float(row["Original_Range_m"]),
            "pairs": pairs.astype(np.int32),
            "bin_id": bins.astype(np.int16),
            "h": h,
            "counts": counts,
            "centers": pair_info["centers"],
            "edges": pair_info["edges"],
            "max_lag": pair_info["max_lag"],
        })

        print(
            f"Directional pair {row['Direction']:8s} "
            f"Az={az0:5.1f}° | pairs={len(pairs):6d}"
        )

    return infos


def experimental_directional_variogram_from_pairs(values_gauss, dir_info):
    """Compute the experimental variogram of one realization in one direction."""
    values = np.asarray(values_gauss, dtype=float)
    pairs = dir_info["pairs"]
    bins = dir_info["bin_id"].astype(int)

    out = np.full(VARIOGRAM_N_LAGS, np.nan, dtype=float)
    if len(pairs) == 0:
        return out

    diff = values[pairs[:, 0]] - values[pairs[:, 1]]
    semi = 0.5 * diff**2
    sums = np.bincount(bins, weights=semi, minlength=VARIOGRAM_N_LAGS)
    counts = np.bincount(bins, minlength=VARIOGRAM_N_LAGS)
    good = counts >= DIRECTIONAL_MIN_PAIRS_PER_BIN
    out[good] = sums[good] / counts[good]
    return out


def hard_data_directional_variograms(hard_coords, hard_gauss, pair_info):
    """
    Recalculate directional experimental variograms from hard data using
    validation lag bins and angular tolerance. This is an independent check;
    the values are identical to the original fitted experimental variogram only if
    the original lag width/tolerance matches these validation settings.
    """
    coords = np.asarray(hard_coords, dtype=float)
    values = np.asarray(hard_gauss, dtype=float)
    n = len(values)

    ii, jj = np.triu_indices(n, k=1)
    dx = coords[jj, 0] - coords[ii, 0]
    dy = coords[jj, 1] - coords[ii, 1]
    h = np.sqrt(dx**2 + dy**2)
    az = pair_azimuth_from_dxdy(dx, dy) % 180.0

    edges = pair_info["edges"]
    max_lag = pair_info["max_lag"]
    base = np.isfinite(h) & (h > EPS) & (h <= max_lag)

    result = {}
    for _, row in DIRECTIONAL_VALIDATION_TABLE.iterrows():
        az0 = float(row["Azimuth_deg"])
        mask = base & (
            axial_angular_difference(az, az0)
            <= (DIRECTIONAL_TOLERANCE_DEG + 1e-12)
        )

        h_d = h[mask]
        ii_d = ii[mask]
        jj_d = jj[mask]
        bins = np.digitize(h_d, edges, right=False) - 1
        valid_bin = (bins >= 0) & (bins < VARIOGRAM_N_LAGS)

        bins = bins[valid_bin]
        ii_d = ii_d[valid_bin]
        jj_d = jj_d[valid_bin]

        curve = np.full(VARIOGRAM_N_LAGS, np.nan, dtype=float)
        counts = np.bincount(bins, minlength=VARIOGRAM_N_LAGS)
        if len(bins) > 0:
            semi = 0.5 * (values[ii_d] - values[jj_d]) ** 2
            sums = np.bincount(
                bins, weights=semi, minlength=VARIOGRAM_N_LAGS
            )
            good = counts >= max(5, DIRECTIONAL_MIN_PAIRS_PER_BIN // 2)
            curve[good] = sums[good] / counts[good]

        result[az0] = {
            "curve": curve,
            "counts": counts,
        }

    return result


def spherical_variogram_with_range(lags, range_m):
    """Spherical variogram with fixed nugget and sill for a specified range."""
    lags = np.asarray(lags, dtype=float)
    r = max(float(range_m), EPS)
    t = lags / r
    gamma = NUGGET + PARTIAL_SILL * spherical_core(t)
    return np.minimum(gamma, SILL)


def fit_directional_range_fixed_nugget_sill(lags, gamma, counts=None, max_lag=None):
    """
    Fit only the range; nugget and sill remain fixed to the SGS model values.
    Used to assess whether ranges from 100 realizations reproduce
    directional range target.
    """
    lags = np.asarray(lags, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    valid = np.isfinite(lags) & np.isfinite(gamma) & (lags > 0)

    if counts is not None:
        counts = np.asarray(counts, dtype=float)
        valid &= counts > 0

    if np.sum(valid) < 4:
        return np.nan

    x = lags[valid]
    y = gamma[valid]

    if counts is None:
        w = np.ones_like(x)
    else:
        c = counts[valid]
        w = np.sqrt(c / max(np.nanmax(c), 1.0))

    upper_base = (
        float(max_lag)
        if max_lag is not None
        else float(np.nanmax(x))
    )
    lower = float(DIRECTIONAL_RANGE_FIT_MIN)
    upper = max(
        lower * 1.5,
        upper_base * DIRECTIONAL_RANGE_FIT_MAX_FACTOR,
        float(np.max(DIRECTIONAL_ORIGINAL_RANGES)) * 1.25,
    )

    def objective(r):
        pred = spherical_variogram_with_range(x, r)
        return float(np.sum(w * (y - pred) ** 2))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(result.x) if result.success else np.nan


def summarize_directional_variogram_validation(
    directional_curves,
    directional_pair_infos,
    hard_directional,
):
    """Directional validation table based on 100 realizations."""
    rows = []
    centers = directional_pair_infos[0]["centers"]

    for model_name in MODELS:
        arr = np.asarray(directional_curves[model_name], dtype=float)

        for d_idx, dir_info in enumerate(directional_pair_infos):
            az0 = dir_info["azimuth_deg"]
            original_range = dir_info["original_range_m"]
            model_range = float(
                get_directional_range(np.array([az0]), model_name)[0]
            )

            curves = arr[:, d_idx, :]
            med = np.nanmedian(curves, axis=0)
            mean_curve = np.nanmean(curves, axis=0)

            target_model = spherical_variogram_with_range(centers, model_range)
            target_original = spherical_variogram_with_range(
                centers, original_range
            )

            valid_model = np.isfinite(med) & np.isfinite(target_model)
            valid_original = np.isfinite(med) & np.isfinite(target_original)

            rmse_med_model = (
                np.sqrt(np.mean((med[valid_model] - target_model[valid_model]) ** 2))
                if np.any(valid_model) else np.nan
            )
            rmse_mean_model = (
                np.sqrt(np.mean((mean_curve[valid_model] - target_model[valid_model]) ** 2))
                if np.any(valid_model) else np.nan
            )
            rmse_med_original = (
                np.sqrt(np.mean((med[valid_original] - target_original[valid_original]) ** 2))
                if np.any(valid_original) else np.nan
            )

            recovered = np.array([
                fit_directional_range_fixed_nugget_sill(
                    centers,
                    curves[r, :],
                    counts=dir_info["counts"],
                    max_lag=dir_info["max_lag"],
                )
                for r in range(N_REALIZATIONS)
            ], dtype=float)

            hard_curve = hard_directional[az0]["curve"]
            hard_counts = hard_directional[az0]["counts"]
            hard_recovered = fit_directional_range_fixed_nugget_sill(
                centers,
                hard_curve,
                counts=hard_counts,
                max_lag=dir_info["max_lag"],
            )

            rows.append({
                "Model": model_name,
                "Direction": dir_info["direction_name"],
                "Azimuth_deg": az0,
                "Original_8dir_Range_m": original_range,
                "Model_Predicted_Range_m": model_range,
                "HardData_Recalc_Range_m": hard_recovered,
                "SimRange_Mean_m": float(np.nanmean(recovered)),
                "SimRange_Median_m": float(np.nanmedian(recovered)),
                "SimRange_P025_m": float(np.nanpercentile(recovered, 2.5)),
                "SimRange_P975_m": float(np.nanpercentile(recovered, 97.5)),
                "RangeBias_Median_vs_Model_m": float(np.nanmedian(recovered) - model_range),
                "RangeBias_Median_vs_Original_m": float(np.nanmedian(recovered) - original_range),
                "Variogram_RMSE_Median_vs_Model": rmse_med_model,
                "Variogram_RMSE_Mean_vs_Model": rmse_mean_model,
                "Variogram_RMSE_Median_vs_Original8dir": rmse_med_original,
                "N_pairs_directional": int(np.sum(dir_info["counts"])),
            })

    return pd.DataFrame(rows)


def save_directional_variogram_tables(
    directional_curves,
    directional_pair_infos,
    hard_directional,
):
    """Save full directional ensemble curves for audit."""
    centers = directional_pair_infos[0]["centers"]

    for model_name in MODELS:
        rows = []
        arr = np.asarray(directional_curves[model_name], dtype=float)

        for d_idx, dir_info in enumerate(directional_pair_infos):
            curves = arr[:, d_idx, :]
            az0 = dir_info["azimuth_deg"]
            model_range = float(
                get_directional_range(np.array([az0]), model_name)[0]
            )
            original_range = dir_info["original_range_m"]

            med = np.nanmedian(curves, axis=0)
            mean_curve = np.nanmean(curves, axis=0)
            lo = np.nanpercentile(curves, 2.5, axis=0)
            hi = np.nanpercentile(curves, 97.5, axis=0)

            hard_curve = hard_directional[az0]["curve"]
            hard_counts = hard_directional[az0]["counts"]

            for k, lag in enumerate(centers):
                rows.append({
                    "Model": model_name,
                    "Direction": dir_info["direction_name"],
                    "Azimuth_deg": az0,
                    "Lag_m": lag,
                    "Grid_pair_count": int(dir_info["counts"][k]),
                    "Hard_pair_count": int(hard_counts[k]),
                    "HardData_Experimental_gamma": hard_curve[k],
                    "Sim_Mean_gamma": mean_curve[k],
                    "Sim_Median_gamma": med[k],
                    "Sim_P025_gamma": lo[k],
                    "Sim_P975_gamma": hi[k],
                    "Target_Model_gamma": spherical_variogram_with_range(
                        np.array([lag]), model_range
                    )[0],
                    "Target_Original8dir_gamma": spherical_variogram_with_range(
                        np.array([lag]), original_range
                    )[0],
                })

        pd.DataFrame(rows).to_csv(
            DIR_ENSEMBLE / f"variogram_directional_8directions_{model_name}.csv",
            index=False,
        )



def plot_recovered_directional_ranges(
    directional_validation_df,
    file_name="18_recovered_directional_range_8directions_2models.png",
):
    """Directly validate eight-direction ranges for both models."""
    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(19, 5.8),
        sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, model_name in zip(axes, MODELS):
        sub = directional_validation_df[
            directional_validation_df["Model"] == model_name
        ].sort_values("Azimuth_deg")

        x = sub["Azimuth_deg"].to_numpy(dtype=float)
        original = sub["Original_8dir_Range_m"].to_numpy(dtype=float)
        model_r = sub["Model_Predicted_Range_m"].to_numpy(dtype=float)
        hard_r = sub["HardData_Recalc_Range_m"].to_numpy(dtype=float)
        med = sub["SimRange_Median_m"].to_numpy(dtype=float)
        lo = sub["SimRange_P025_m"].to_numpy(dtype=float)
        hi = sub["SimRange_P975_m"].to_numpy(dtype=float)

        ax.plot(
            x, original, marker="o", linewidth=1.8,
            label="Input eight-direction range"
        )
        ax.plot(
            x, model_r, marker="s", linewidth=1.5, linestyle="--",
            label="Range model NLS"
        )
        ax.plot(
            x, hard_r, marker="x", linewidth=1.2, linestyle=":",
            label="Range hard data (recalc)"
        )
        ax.errorbar(
            x,
            med,
            yerr=np.vstack([med - lo, hi - med]),
            fmt="^",
            capsize=3,
            linewidth=1.0,
            label="Median simulated range + 95% interval",
        )

        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Azimuth (°)")
        ax.set_xticks(DIRECTIONAL_AZIMUTHS)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Directional range (m)")
    fig.suptitle(
        "Validation of Eight-Direction Range Reproduction from 100 SGS Realizations",
        fontsize=13, fontweight="bold"
    )

    if SAVE_FIGS:
        fig.savefig(
            output_file(Path("05_ensemble_diagnostics") / file_name),
            dpi=MAP_DPI,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_directional_variogram_validation(
    directional_curves,
    directional_pair_infos,
    hard_directional,
):
    """
    Create one 2x4 figure for each model.
    Each directional panel displays 100 realization variogram curves so that
    reproduction across realizations is visible. No median or
    target-model line is shown because each direction has a different range.
    """
    centers = directional_pair_infos[0]["centers"]
    x_max = directional_pair_infos[0]["max_lag"]
    y_max = max(1.25 * SILL, 1.15)

    for model_name in MODELS:
        arr = np.asarray(directional_curves[model_name], dtype=float)
        fig, axes = plt.subplots(
            2, 4, figsize=(20, 10),
            sharex=True, sharey=True, constrained_layout=True
        )
        axes = axes.ravel()

        for d_idx, (ax, dir_info) in enumerate(zip(axes, directional_pair_infos)):
            curves = arr[:, d_idx, :]
            az0 = dir_info["azimuth_deg"]
            hard_curve = hard_directional[az0]["curve"]

            for r in range(curves.shape[0]):
                ax.plot(
                    centers, curves[r],
                    linewidth=0.65, alpha=0.14,
                    label="100 realization" if r == 0 else None,
                )

            hard_valid = np.isfinite(hard_curve)
            ax.scatter(
                centers[hard_valid],
                hard_curve[hard_valid],
                s=22, marker="x",
                label="Experimental hard data (recalc)",
            )
            ax.axhline(SILL, linewidth=0.8, linestyle="-.", label="Sill" if d_idx == 0 else None)

            ax.set_title(
                f"{dir_info['direction_name']} | Az {az0:.1f}°",
                fontweight="bold"
            )
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.grid(alpha=0.2)

        for ax in axes[4:]:
            ax.set_xlabel("Lag distance (m)")
        axes[0].set_ylabel("Normal-score semivariance")
        axes[4].set_ylabel("Normal-score semivariance")
        axes[0].legend(fontsize=7, loc="best")

        fig.suptitle(
            f"Eight-Direction Variograms - 100 SGS Realizations {model_name}",
            fontsize=14, fontweight="bold"
        )
        if SAVE_FIGS:
            fig.savefig(
                DIR_ENSEMBLE /
                f"17_variogram_directional_8directions_validation_{model_name}.png",
                dpi=MAP_DPI,
                bbox_inches="tight",
            )

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def histogram_validation_table(realizations, z_original):
    """Summarize global-distribution reproduction for histogram validation."""
    z_original = np.asarray(z_original, dtype=float)
    z_original = z_original[np.isfinite(z_original)]

    raw_stats = {
        "Mean": np.mean(z_original),
        "Std": np.std(z_original, ddof=1),
        "P10": np.percentile(z_original, 10),
        "P25": np.percentile(z_original, 25),
        "P50": np.percentile(z_original, 50),
        "P75": np.percentile(z_original, 75),
        "P90": np.percentile(z_original, 90),
    }

    rows = []
    for model_name in MODELS:
        arr = np.asarray(realizations[model_name], dtype=float)
        pooled = arr[np.isfinite(arr)]

        model_stats = {
            "Mean": np.mean(pooled),
            "Std": np.std(pooled, ddof=1),
            "P10": np.percentile(pooled, 10),
            "P25": np.percentile(pooled, 25),
            "P50": np.percentile(pooled, 50),
            "P75": np.percentile(pooled, 75),
            "P90": np.percentile(pooled, 90),
        }

        # Quantile RMSE provides a simple summary of distributional similarity without
        
        qkeys = ["P10", "P25", "P50", "P75", "P90"]
        q_rmse = np.sqrt(np.mean([
            (model_stats[k] - raw_stats[k]) ** 2 for k in qkeys
        ]))

        per_real_mean = np.nanmean(arr, axis=1)
        per_real_std = np.nanstd(arr, axis=1, ddof=1)

        rows.append({
            "Model": model_name,
            "Raw_Mean": raw_stats["Mean"],
            "Sim_Pooled_Mean": model_stats["Mean"],
            "Mean_Bias": model_stats["Mean"] - raw_stats["Mean"],
            "Raw_Std": raw_stats["Std"],
            "Sim_Pooled_Std": model_stats["Std"],
            "Std_Ratio_Sim_to_Raw": model_stats["Std"] / max(raw_stats["Std"], EPS),
            "Raw_P10": raw_stats["P10"],
            "Sim_P10": model_stats["P10"],
            "Raw_P25": raw_stats["P25"],
            "Sim_P25": model_stats["P25"],
            "Raw_P50": raw_stats["P50"],
            "Sim_P50": model_stats["P50"],
            "Raw_P75": raw_stats["P75"],
            "Sim_P75": model_stats["P75"],
            "Raw_P90": raw_stats["P90"],
            "Sim_P90": model_stats["P90"],
            "Quantile_RMSE": q_rmse,
            "Mean_of_100_realization_means": float(np.nanmean(per_real_mean)),
            "SD_of_100_realization_means": float(np.nanstd(per_real_mean, ddof=1)),
            "Mean_of_100_realization_stds": float(np.nanmean(per_real_std)),
            "SD_of_100_realization_stds": float(np.nanstd(per_real_std, ddof=1)),
        })

    return pd.DataFrame(rows)


def _global_thickness_range(realizations, z_original):
    chunks = [np.asarray(z_original, dtype=float)]
    for model_name in MODELS:
        chunks.append(np.asarray(realizations[model_name], dtype=float).ravel())
    values = np.concatenate([v[np.isfinite(v)] for v in chunks])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_histogram_100_realizations_by_model(realizations, z_original):
    """
    Create one figure per model. Each figure displays 100 histogram lines
    (step histograms) to preserve the original histogram shape without
    excessive smoothing. No mean line is shown.
    """
    vmin, vmax = _global_thickness_range(realizations, z_original)
    bins = np.linspace(vmin, vmax, HISTOGRAM_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    raw = np.asarray(z_original, dtype=float)
    raw = raw[np.isfinite(raw)]
    raw_counts, _ = np.histogram(raw, bins=bins)
    raw_rel = raw_counts.astype(float) / max(1, raw_counts.sum())

    for model_name in MODELS:
        arr = np.asarray(realizations[model_name], dtype=float)
        fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)

        for r in range(arr.shape[0]):
            vals = arr[r]
            vals = vals[np.isfinite(vals)]
            counts, _ = np.histogram(vals, bins=bins)
            rel = counts.astype(float) / max(1, counts.sum())
            ax.step(
                centers, rel, where="mid",
                linewidth=0.7, alpha=0.16,
                label="100 realization" if r == 0 else None,
            )

        ax.step(
            centers, raw_rel, where="mid",
            linewidth=2.0, linestyle="--",
            label="Drillhole-data histogram",
        )

        ax.set_title(
            f"Histogram 100 Realization SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Limonite Thickness")
        ax.set_ylabel("Relative frequency")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_ensemble_diagnostics") /
                    f"13_histogram_100_realization_{model_name.lower()}.png"
                ),
                dpi=MAP_DPI,
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def plot_uncertainty_ensemble(post, file_name="14_uncertainty_plot_100_realization_2model.png"):
    """Create one uncertainty-plot figure from 100 realizations for both models."""
    reference = np.mean(np.vstack([post[m]["Etype"] for m in MODELS]), axis=0)
    order = np.argsort(reference)
    x = np.arange(1, len(order) + 1)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(20, 6), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, model_name in zip(axes, MODELS):
        p = post[model_name]
        lo = p["CL95_Lower"][order]
        hi = p["CL95_Upper"][order]
        p25 = p["P25"][order]
        p75 = p["P75"][order]
        p50 = p["P50"][order]
        et = p["Etype"][order]
        ax.fill_between(x, lo, hi, alpha=0.18, label="P2.5-P97.5")
        ax.fill_between(x, p25, p75, alpha=0.28, label="P25-P75")
        ax.plot(x, p50, linewidth=1.25, label="P50")
        ax.plot(x, et, linewidth=1.0, linestyle="--", label="E-type")
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Grid node (reference E-type order)")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Limonite Thickness")
    fig.suptitle("SGS uncertainty plot - 100 realizations and simulation interval per node", fontsize=13, fontweight="bold")

    if SAVE_FIGS:
        fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _empirical_cdf_on_grid(values, x_grid):
    vals = np.asarray(values, dtype=float)
    vals = np.sort(vals[np.isfinite(vals)])
    if len(vals) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    return np.searchsorted(vals, x_grid, side="right") / float(len(vals))


def plot_cdf_100_realizations_by_model(realizations, z_original):
    """Create one CDF figure per model showing 100 realization curves."""
    vmin, vmax = _global_thickness_range(realizations, z_original)
    x = np.linspace(vmin, vmax, CDF_BINS)
    raw_cdf = _empirical_cdf_on_grid(z_original, x)

    for model_name in MODELS:
        arr = np.asarray(realizations[model_name], dtype=float)
        curves = []
        fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)

        for r in range(arr.shape[0]):
            cdf = _empirical_cdf_on_grid(arr[r], x)
            curves.append(cdf)
            ax.plot(
                x, cdf,
                linewidth=0.65, alpha=0.15,
                label="100 realization" if r == 0 else None,
            )

        curves = np.asarray(curves, dtype=float)
        mean_cdf = np.nanmean(curves, axis=0)
        ax.plot(x, mean_cdf, linewidth=2.4, label="Mean CDF of 100 realizations")
        ax.plot(x, raw_cdf, linewidth=2.0, linestyle="--", label="Drillhole-data CDF")

        ax.set_title(
            f"CDF 100 Realization SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Limonite Thickness")
        ax.set_ylabel("Probabilitas kumulatif")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_ensemble_diagnostics") /
                    f"15_CDF_100_realization_{model_name.lower()}.png"
                ),
                dpi=MAP_DPI,
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def plot_variogram_100_realizations_by_model(variogram_curves, pair_info):
    """Create one variogram figure per model showing 100 realization curves."""
    centers = pair_info["centers"]

    for model_name in MODELS:
        curves = np.asarray(variogram_curves[model_name], dtype=float)
        target = theoretical_omnidirectional_variogram(centers, model_name)
        mean_curve = np.nanmean(curves, axis=0)
        median_curve = np.nanmedian(curves, axis=0)

        fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)

        for r in range(curves.shape[0]):
            ax.plot(
                centers, curves[r],
                linewidth=0.7, alpha=0.16,
                label="100 variogram realization" if r == 0 else None,
            )

        ax.plot(
            centers, mean_curve,
            linewidth=2.4, marker="o", markersize=3,
            label="Mean 100 realization",
        )
        ax.plot(
            centers, median_curve,
            linewidth=1.8, linestyle="--",
            label="Median 100 realization",
        )
        ax.plot(
            centers, target,
            linewidth=2.2, linestyle=":",
            label="Variogram target",
        )
        ax.axhline(SILL, linewidth=1.0, linestyle="-.", label="Sill")

        ax.set_title(
            f"Variogram 100 Realization SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Normal-score semivariance")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_ensemble_diagnostics") /
                    f"16_variogram_100_realization_{model_name.lower()}.png"
                ),
                dpi=MAP_DPI,
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# 15. MAIN SGS
# ============================================================

def main():
    start_all = time.time()
    prepare_output_folders()

    print("\n" + "=" * 72)
    print("OUTPUT DIRECTORIES")
    print("=" * 72)
    print(f"Root output            = {OUTPUT_DIR.resolve()}")
    print(f"Realization CSV files = {DIR_REALIZATION_CSV.resolve()}")
    print(f"Individual maps = {DIR_INDIVIDUAL_MAPS.resolve()}")
    print(f"Comparison maps = {DIR_COMPARE_MAPS.resolve()}")
    print(f"Summary/diagnostics   = {DIR_SUMMARY.resolve()}")

    data = load_thickness_data()
    hard_coords = data[["X", "Y"]].values.astype(float)
    z_original = data["Z"].values.astype(float)

    
    hard_gauss, y_table, z_table = normal_score_transform(z_original)

    print("\n" + "=" * 72)
    print("NORMAL SCORE")
    print("=" * 72)
    print(f"Mean normal score = {np.mean(hard_gauss):.6f}")
    print(f"SD normal score = {np.std(hard_gauss, ddof=1):.6f}")
    print(f"Nugget = {NUGGET:.4f}")
    print(f"Partial sill = {PARTIAL_SILL:.4f}")
    print(f"Total sill = {SILL:.4f}")

    print("\n" + "=" * 72)
    print("DIRECTIONAL RANGE PARAMETERS")
    print("=" * 72)
    print(
        f"Ellipse   : Az={ELLIPSE_PHI_AZ:.5f}°, "
        f"A={ELLIPSE_MAJOR:.5f} m, B={ELLIPSE_MINOR:.5f} m, "
        f"A/B={ELLIPSE_RATIO:.5f}"
    )
    print(
        f"Cassini   : Az={CASSINI_PHI_AZ:.5f}°, "
        f"a={CASSINI_A:.5f} m, b={CASSINI_B:.5f} m"
    )
    print(f"NMIN={NMIN}, NMAX={NMAX}, search h/R={NORMALIZED_SEARCH_RADIUS}")
    print(f"Number of realizations = {N_REALIZATIONS}")
    print(f"Base seed = {BASE_SEED}")

    grid_info = build_grid(data)
    valid_points = grid_info["valid_points"]
    n_valid = len(valid_points)

    # Candidate cache per model.
    caches = {
        model_name: build_candidate_cache(valid_points, hard_coords, model_name)
        for model_name in MODELS
    }


    cv_summary_rows = []
    cv_details = {}
    if RUN_LOOCV:
        print("\n" + "=" * 72)
        print("LOOCV SIMPLE KRIGING - NORMAL SCORE")
        print("=" * 72)
        for model_name in MODELS:
            summary, detail = loocv_model(
                hard_coords, hard_gauss, z_original, y_table, z_table, model_name
            )
            cv_summary_rows.append(summary)
            cv_details[model_name] = detail
            print(
                f"{model_name:10s} | RMSE={summary['CV_RMSE']:.4f} | "
                f"MAE={summary['CV_MAE']:.4f} | "
                f"ME={summary['CV_ME']:.4f} | R2={summary['CV_R2']:.4f}"
            )

    
    realizations = {
        model_name: np.empty((N_REALIZATIONS, n_valid), dtype=np.float32)
        for model_name in MODELS
    }

    
    if RUN_ENSEMBLE_DIAGNOSTICS:
        pair_info = build_variogram_pair_sample(valid_points)
        variogram_curves = {
            model_name: np.full(
                (N_REALIZATIONS, VARIOGRAM_N_LAGS),
                np.nan,
                dtype=np.float32,
            )
            for model_name in MODELS
        }

        if RUN_DIRECTIONAL_VARIOGRAM_VALIDATION:
            directional_pair_infos = build_directional_pair_infos(pair_info)
            hard_directional = hard_data_directional_variograms(
                hard_coords, hard_gauss, pair_info
            )
            directional_variogram_curves = {
                model_name: np.full(
                    (
                        N_REALIZATIONS,
                        len(DIRECTIONAL_VALIDATION_TABLE),
                        VARIOGRAM_N_LAGS,
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                for model_name in MODELS
            }
        else:
            directional_pair_infos = None
            hard_directional = None
            directional_variogram_curves = None
    else:
        pair_info = None
        variogram_curves = None
        directional_pair_infos = None
        hard_directional = None
        directional_variogram_curves = None

    diagnostics_rows = []

    print("\n" + "=" * 72)
    print(f"RUNNING SGS: {N_REALIZATIONS} REALIZATIONS - 2 MODELS")
    print("=" * 72)
    print(
        "Each paired Ellipse/Cassini realization uses "
        "the same random path and Gaussian innovations."
    )

    for r in range(N_REALIZATIONS):
        realization_number = r + 1
        seed = BASE_SEED + r
        rng = np.random.default_rng(seed)

        
        random_path = rng.permutation(n_valid)
        gaussian_innovations = rng.standard_normal(n_valid)
        t0 = time.time()

        for model_name in MODELS:
            sim_gauss, diag = simulate_one_realization(
                model_name,
                valid_points,
                hard_coords,
                hard_gauss,
                caches[model_name],
                random_path,
                gaussian_innovations,
            )

            if RUN_ENSEMBLE_DIAGNOSTICS:
                variogram_curves[model_name][r, :] = experimental_variogram_from_pairs(
                    sim_gauss, pair_info
                ).astype(np.float32)

                if RUN_DIRECTIONAL_VARIOGRAM_VALIDATION:
                    for d_idx, dir_info in enumerate(directional_pair_infos):
                        directional_variogram_curves[model_name][r, d_idx, :] = (
                            experimental_directional_variogram_from_pairs(
                                sim_gauss, dir_info
                            ).astype(np.float32)
                        )

            sim_z = back_transform(
                sim_gauss, y_table, z_table, BACKTRANSFORM_TAIL
            )
            realizations[model_name][r, :] = sim_z.astype(np.float32)

            diagnostics_rows.append({
                "Realization": realization_number,
                "Seed": seed,
                "Model": model_name,
                "N_local_repairs": diag["n_repaired"],
                "Max_jitter": diag["max_jitter"],
                "N_exact_hard_nodes": diag["n_exact_hard"],
                "Sim_mean": float(np.mean(sim_z)),
                "Sim_std": float(np.std(sim_z, ddof=1)),
                "Sim_min": float(np.min(sim_z)),
                "Sim_max": float(np.max(sim_z)),
            })

        
        
        if SAVE_EACH_REALIZATION_CSV and SAVE_REALIZATION_CSV_DURING_SIMULATION:
            csv_path = DIR_REALIZATION_CSV / (
                f"R{realization_number:03d}_SGS_LIM_Ellipse_Cassini.csv"
            )
            pd.DataFrame({
                "X": valid_points[:, 0],
                "Y": valid_points[:, 1],
                **{f"SGS_{m}": realizations[m][r] for m in MODELS},
            }).to_csv(csv_path, index=False)

        
        if realization_number % 5 == 0 or realization_number == N_REALIZATIONS:
            pd.DataFrame(diagnostics_rows).to_csv(
                DIR_TABLES / "diagnostic_checkpoint.csv", index=False
            )

        dt = time.time() - t0
        print(
            f"Realization {realization_number:03d}/{N_REALIZATIONS} completed "
            f"| seed={seed} | {dt:.1f} detik"
        )

    diagnostics_df = pd.DataFrame(diagnostics_rows)

    # ---------------- POST-PROCESSING ----------------
    post = {
        model_name: postprocess_realizations(realizations[model_name])
        for model_name in MODELS
    }

    directional_validation_df = None
    if (
        RUN_ENSEMBLE_DIAGNOSTICS
        and RUN_DIRECTIONAL_VARIOGRAM_VALIDATION
        and directional_variogram_curves is not None
    ):
        directional_validation_df = summarize_directional_variogram_validation(
            directional_variogram_curves,
            directional_pair_infos,
            hard_directional,
        )

    # ---------------- SUMMARY COMPARISON ----------------
    comparison_rows = []
    cv_lookup = {row["Model"]: row for row in cv_summary_rows} if cv_summary_rows else {}

    for model_name in MODELS:
        row = {
            "Model": model_name,
            "Directional_fit_RMSE_m": DIRECTIONAL_FIT_RMSE[model_name],
            "Directional_fit_R2": DIRECTIONAL_FIT_R2[model_name],
            "HardData_mean": float(np.mean(z_original)),
            "HardData_std": float(np.std(z_original, ddof=1)),
            "Mean_of_realization_means": float(
                diagnostics_df.loc[diagnostics_df["Model"] == model_name, "Sim_mean"].mean()
            ),
            "Mean_of_realization_stds": float(
                diagnostics_df.loc[diagnostics_df["Model"] == model_name, "Sim_std"].mean()
            ),
            "Mean_Etype": float(np.mean(post[model_name]["Etype"])),
            "Mean_ensemble_SD": float(np.mean(post[model_name]["SD"])),
            "Mean_CL95_width": float(np.mean(post[model_name]["CL95_Width"])),
            "Total_local_repairs": int(
                diagnostics_df.loc[diagnostics_df["Model"] == model_name, "N_local_repairs"].sum()
            ),
            "Max_jitter_overall": float(
                diagnostics_df.loc[diagnostics_df["Model"] == model_name, "Max_jitter"].max()
            ),
        }

        if directional_validation_df is not None:
            dsub = directional_validation_df[
                directional_validation_df["Model"] == model_name
            ]
            row.update({
                "DirVario_Mean_RMSE_Median_vs_Model": float(
                    dsub["Variogram_RMSE_Median_vs_Model"].mean()
                ),
                "DirRange_MeanAbsBias_Median_vs_Model_m": float(
                    np.mean(np.abs(dsub["RangeBias_Median_vs_Model_m"]))
                ),
                "DirRange_MeanAbsBias_Median_vs_Original8dir_m": float(
                    np.mean(np.abs(dsub["RangeBias_Median_vs_Original_m"]))
                ),
            })

        if model_name in cv_lookup:
            row.update({
                "CV_RMSE": cv_lookup[model_name]["CV_RMSE"],
                "CV_MAE": cv_lookup[model_name]["CV_MAE"],
                "CV_ME": cv_lookup[model_name]["CV_ME"],
                "CV_R2": cv_lookup[model_name]["CV_R2"],
            })
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # ---------------- SAVE OUTPUT ----------------
    diagnostics_df.to_csv(DIR_TABLES / "diagnostic_100_realization_2model.csv", index=False)
    comparison_df.to_csv(DIR_TABLES / "comparison_summary_Ellipse_Cassini.csv", index=False)

    if RUN_LOOCV:
        pd.DataFrame(cv_summary_rows).to_csv(DIR_TABLES / "LOOCV_summary.csv", index=False)
        for model_name in MODELS:
            cv_details[model_name].to_csv(
                DIR_TABLES / f"LOOCV_detail_{model_name}.csv", index=False
            )

    if SAVE_ALL_REALIZATIONS:
        for model_name in MODELS:
            np.save(
                DIR_ARRAYS / f"SGS_LIM_{model_name}_{N_REALIZATIONS}realization.npy",
                realizations[model_name],
            )
        np.savez(
            DIR_ARRAYS / "grid_metadata_SGS_LIM.npz",
            x_grid=grid_info["x_grid"],
            y_grid=grid_info["y_grid"],
            inside_mask=grid_info["inside_mask"],
            valid_flat_idx=grid_info["valid_flat_idx"],
            grid_shape=np.array(grid_info["XX"].shape, dtype=int),
        )

    if SAVE_POSTPROCESS_CSV:
        for model_name in MODELS:
            out = pd.DataFrame({
                "X": valid_points[:, 0],
                "Y": valid_points[:, 1],
                "Etype": post[model_name]["Etype"],
                "SD": post[model_name]["SD"],
                "Variance": post[model_name]["Variance"],
                "CL95_Lower": post[model_name]["CL95_Lower"],
                "CL95_Upper": post[model_name]["CL95_Upper"],
                "CL95_Width": post[model_name]["CL95_Width"],
                "P25": post[model_name]["P25"],
                "P50": post[model_name]["P50"],
                "P75": post[model_name]["P75"],
                "P95": post[model_name]["P95"],
            })
            out.to_csv(DIR_TABLES / f"postprocess_LIM_{model_name}.csv", index=False)

    pd.DataFrame({
        "NormalScore_table": y_table,
        "Thickness_LIM_sorted": z_table,
    }).to_csv(DIR_TABLES / "normal_score_backtransform_table.csv", index=False)

    
    histogram_validation_df = histogram_validation_table(
        realizations, z_original
    )
    histogram_validation_df.to_csv(
        DIR_ENSEMBLE / "histogram_validation_summary.csv",
        index=False,
    )

    if RUN_ENSEMBLE_DIAGNOSTICS:
        
        vario_table = pd.DataFrame({"Lag_m": pair_info["centers"]})
        for model_name in MODELS:
            curves = variogram_curves[model_name]
            vario_table[f"{model_name}_Mean"] = np.nanmean(curves, axis=0)
            vario_table[f"{model_name}_Median"] = np.nanmedian(curves, axis=0)
            vario_table[f"{model_name}_P025"] = np.nanpercentile(curves, 2.5, axis=0)
            vario_table[f"{model_name}_P975"] = np.nanpercentile(curves, 97.5, axis=0)
            vario_table[f"{model_name}_Target"] = theoretical_omnidirectional_variogram(
                pair_info["centers"], model_name
            )
        vario_table["Pair_count"] = pair_info["counts"]
        vario_table.to_csv(
            DIR_ENSEMBLE / "variogram_ensemble_100_realization.csv",
            index=False,
        )

        if RUN_DIRECTIONAL_VARIOGRAM_VALIDATION:
            DIRECTIONAL_VALIDATION_TABLE.to_csv(
                DIR_TABLES / "directional_range_8directions_input.csv",
                index=False,
            )
            directional_validation_df.to_csv(
                DIR_ENSEMBLE / "directional_variogram_validation_summary.csv",
                index=False,
            )
            save_directional_variogram_tables(
                directional_variogram_curves,
                directional_pair_infos,
                hard_directional,
            )

    
    global_vmin = float(np.nanmin(z_original))
    global_vmax = float(np.nanmax(z_original))
    for m in MODELS:
        global_vmin = min(
            global_vmin,
            float(np.nanmin(realizations[m])),
        )
        global_vmax = max(
            global_vmax,
            float(np.nanmax(realizations[m])),
        )

    print(
        f"Global thickness-map scale = "
        f"{global_vmin:.4f} s.d. {global_vmax:.4f}"
    )

    print("\n" + "=" * 72)
    print("SAVING 100 REALIZATIONS AND SGS MAPS FOR BOTH MODELS")
    print("=" * 72)
    print(f"Output root folder = {OUTPUT_DIR.resolve()}")

    for r in range(N_REALIZATIONS):
        real_no = r + 1
        maps_r = {
            m: valid_to_full_grid(realizations[m][r], grid_info)
            for m in MODELS
        }

        if SAVE_EACH_REALIZATION_CSV and not SAVE_REALIZATION_CSV_DURING_SIMULATION:
            realization_df = pd.DataFrame({
                "X": valid_points[:, 0],
                "Y": valid_points[:, 1],
                **{f"SGS_{m}": realizations[m][r] for m in MODELS},
            })
            realization_df.to_csv(
                DIR_REALIZATION_CSV / f"R{real_no:03d}_SGS_LIM_Ellipse_Cassini.csv",
                index=False,
            )

        if SAVE_INDIVIDUAL_REALIZATION_MAPS:
            for model_name in MODELS:
                rel_file = Path("02_map_individual") / model_name / (
                    f"R{real_no:03d}_Map_SGS_LIM_{model_name}.png"
                )
                if should_export(rel_file):
                    plot_single_sgs_map(
                        maps_r[model_name],
                        grid_info,
                        hard_coords,
                        z_original,
                        model_name,
                        real_no,
                        rel_file,
                        global_vmin=global_vmin,
                        global_vmax=global_vmax,
                    )

        if SAVE_PAIRED_REALIZATION_MAPS:
            rel_compare = Path("03_comparison_maps") / (
                f"R{real_no:03d}_Comparison_Map_2Models.png"
            )
            if should_export(rel_compare):
                plot_comparison_sgs_map(
                    maps_r,
                    grid_info,
                    hard_coords,
                    z_original,
                    real_no,
                    rel_compare,
                    global_vmin,
                    global_vmax,
                )

        
        plt.close("all")
        if real_no % 5 == 0:
            gc.collect()

        if real_no % 10 == 0 or real_no == 1:
            print(f"  exported/verified realization map {real_no:03d}/{N_REALIZATIONS}")

    # ---------------- SPATIAL SUMMARY PLOTS ----------------
    example_idx = int(np.clip(EXAMPLE_REALIZATION - 1, 0, N_REALIZATIONS - 1))
    example_maps = {
        m: valid_to_full_grid(realizations[m][example_idx], grid_info)
        for m in MODELS
    }
    plot_model_maps(
        example_maps,
        grid_info,
        hard_coords,
        {m: f"{m} - Realization {example_idx + 1}" for m in MODELS},
        "SGS Limonite Thickness - Example Paired Realization for Two Models",
        Path("04_spatial_summary") / "01_example_realization_2models.png",
        "Limonite Thickness",
        fixed_vmin=global_vmin,
        fixed_vmax=global_vmax,
    )

    summary_specs = [
        ("Etype", "E-type", "E-type of 100 Realization SGS - Limonite Thickness", "02_Etype_2model.png", "Limonite Thickness"),
        ("SD", "SD", "Standard Deviation 100 Realization SGS - Limonite Thickness", "03_SD_2model.png", "SD Limonite Thickness"),
        ("Variance", "Variance", "Variance 100 Realization SGS - Limonite Thickness", "04_Variance_2model.png", "Variance"),
        ("CL95_Lower", "P2.5", "Lower 95% Simulation Interval SGS - Limonite Thickness", "05_P025_2model.png", "Limonite Thickness"),
        ("CL95_Upper", "P97.5", "Upper 95% Simulation Interval SGS - Limonite Thickness", "06_P975_2model.png", "Limonite Thickness"),
        ("CL95_Width", "Width P2.5-P97.5", "Width 95% Simulation Interval SGS - Limonite Thickness", "07_interval95_width_2model.png", "Interval width"),
        ("P25", "P25", "P25 of 100 Realization SGS - Limonite Thickness", "08_P25_2model.png", "Limonite Thickness"),
        ("P50", "P50", "P50 of 100 Realization SGS - Limonite Thickness", "09_P50_2model.png", "Limonite Thickness"),
        ("P75", "P75", "P75 of 100 Realization SGS - Limonite Thickness", "10_P75_2model.png", "Limonite Thickness"),
        ("P95", "P95", "P95 of 100 Realization SGS - Limonite Thickness", "11_P95_2model.png", "Limonite Thickness"),
    ]

    
    
    
    thickness_keys = {
        "Etype", "CL95_Lower", "CL95_Upper",
        "P25", "P50", "P75", "P95",
    }

    etype_maps = None
    for key, label, suptitle, filename, cbar_label in summary_specs:
        maps = {
            m: valid_to_full_grid(post[m][key], grid_info)
            for m in MODELS
        }
        titles = {m: f"{label} {m}" for m in MODELS}
        plot_model_maps(
            maps,
            grid_info,
            hard_coords,
            titles,
            suptitle,
            Path("04_spatial_summary") / filename,
            cbar_label,
            fixed_vmin=global_vmin if key in thickness_keys else None,
            fixed_vmax=global_vmax if key in thickness_keys else None,
        )
        if key == "Etype":
            etype_maps = maps

    if etype_maps is not None:
        plot_etype_differences(
            etype_maps,
            grid_info,
            hard_coords,
            Path("04_spatial_summary") / "12_difference_Etype_2model.png",
        )

    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model_name in MODELS:
        sub = diagnostics_df[diagnostics_df["Model"] == model_name]
        axes[0].hist(sub["Sim_mean"], bins=15, alpha=0.45, label=model_name)
        axes[1].hist(sub["Sim_std"], bins=15, alpha=0.45, label=model_name)
    axes[0].axvline(np.mean(z_original), linestyle="--", linewidth=1.5, label="Hard data mean")
    axes[1].axvline(np.std(z_original, ddof=1), linestyle="--", linewidth=1.5, label="Hard data std")
    axes[0].set_title("Mean per Realization")
    axes[1].set_title("Standard Deviation per Realization")
    axes[0].set_xlabel("Mean Limonite Thickness")
    axes[1].set_xlabel("SD Limonite Thickness")
    axes[0].set_ylabel("Frequency")
    axes[1].set_ylabel("Frequency")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    if SAVE_FIGS:
        fig.savefig(DIR_ENSEMBLE / "12b_histogram_mean_std_per_realization.png", dpi=250, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    
    if RUN_ENSEMBLE_DIAGNOSTICS:
        plot_histogram_100_realizations_by_model(
            realizations,
            z_original,
        )
        plot_uncertainty_ensemble(
            post,
            Path("05_ensemble_diagnostics") /
            "14_uncertainty_plot_100_realization_2model.png",
        )
        plot_cdf_100_realizations_by_model(
            realizations,
            z_original,
        )
        plot_variogram_100_realizations_by_model(
            variogram_curves,
            pair_info,
        )

        if RUN_DIRECTIONAL_VARIOGRAM_VALIDATION:
            plot_directional_variogram_validation(
                directional_variogram_curves,
                directional_pair_infos,
                hard_directional,
            )
            plot_recovered_directional_ranges(
                directional_validation_df
            )

    elapsed = time.time() - start_all
    print("\n" + "=" * 72)
    print("SGS COMPLETE")
    print("=" * 72)
    print(comparison_df.round(5).to_string(index=False))
    print(f"\nWaktu total = {elapsed / 60.0:.2f} menit")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("\nOutput folder structure:")
    print(f"- 01_realization_csv      : {DIR_REALIZATION_CSV}")
    print(f"- 02_map_individual    : {DIR_INDIVIDUAL_MAPS}")
    print(f"- 03_comparison_maps  : {DIR_COMPARE_MAPS}")
    print(f"- 04_spatial_summary  : {DIR_SUMMARY}")
    print(f"- 05_ensemble_diagnostics: {DIR_ENSEMBLE}")
    print(f"- 06_diagnostic_tables   : {DIR_TABLES}")
    print(f"- 07_array_npy          : {DIR_ARRAYS}")
    print("\nMain outputs:")
    print(f"1. {N_REALIZATIONS} CSV paired 2 model (R001 ... R{N_REALIZATIONS:03d})")
    print(f"2. {N_REALIZATIONS} SGS maps - Ellipse")
    print(f"3. {N_REALIZATIONS} map SGS Cassini")
    print(f"4. {N_REALIZATIONS} paired comparison maps for two models")
    print("5. E-type, SD, variance, simulation intervals, percentiles, and Cassini-Ellipse difference map")
    print("6. Histogram: 100 realization step curves per model (Y = relative frequency)")
    print("7. Uncertainty plot 100 realization")
    print("8. CDF: 100 realization curves in one figure for each model")
    print("9. Variogram: 100 realization curves in one figure for each model")
    print("10. Eight-direction variograms: two figures (Ellipse, Cassini)")
    print("11. Table of recovered directional ranges and variogram RMSE")
    print("12. Recovered eight-direction range plot (input vs model vs simulation)")

    return {
        "comparison": comparison_df,
        "diagnostics": diagnostics_df,
        "postprocess": post,
        "realizations": realizations,
        "variogram_curves": variogram_curves,
        "directional_variogram_curves": directional_variogram_curves,
        "directional_validation": directional_validation_df,
        "histogram_validation": histogram_validation_df,
    }


if __name__ == "__main__":
    main()