
# SEQUENTIAL GAUSSIAN SIMULATION (SGS) OF LIMONITE THICKNESS
# COMPARISON OF ELLIPSE VS CASSINI OVAL
# ============================================================
#
# Input:
#   Ketebalan.xlsx
#   Kolom variabel: "Ketebalan LIM"
#   Kolom koordinat X/Y dideteksi otomatis (X/Easting dan Y/Northing).
#
# Konsep:
#   1. Normal-score transform Ketebalan LIM.
#   2. SGS pada ruang Gaussian menggunakan Simple Kriging (mean = 0).
#   3. Dua model memakai parameter variogram radial yang SAMA:
#        nugget = 0.06
#        partial sill = 0.94
#        total sill = 1.00
#        model = spherical
#   4. Yang dibedakan adalah geometri directional range:
#        - Elips
#        - Oval Cassini affine-scaled
#   5. 100 realisasi menggunakan paired random path dan paired Gaussian
#      innovations to support a paired comparison of both geometries.
#   6. Back-transform hasil SGS ke satuan ketebalan asli.
#   7. Menyimpan 100 realisasi ke ROOT OUTPUT BARU dan subfolder terpisah.
#      CSV disimpan segera setiap realisasi selesai; peta dipisahkan per model.
#
# Grid / neighborhood:
#   GRID_RES = 12.5 m
#   boundary ortogonal rapi mengikuti sebaran titik bor
#   buffer luar = 1 x spasi titik bor = 25 m
#   lubang internal boundary diisi
#   boundary memakai raster ortogonal 25 m agar rapi seperti batas IUP
#   NMIN = 6
#   NMAX = 12
#   NORMALIZED_SEARCH_RADIUS = 1.25
#
# Dependensi Anaconda:
#   numpy, pandas, scipy, matplotlib, openpyxl
#
# Catatan metodologi:
#   - SGS menggunakan Simple Kriging karena variabel telah ditransformasikan
#     ke normal score dengan mean teoritis 0 dan sill 1.
#   - Cassini memakai directional range yang sama dengan formulasi pada
#     script estimasi. Karena geometri Cassini bukan anisotropi Euclidean
#     standar, script melakukan stabilisasi numerik lokal bila matriks
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
matplotlib.use("Agg")  # lebih stabil untuk menyimpan ratusan peta tanpa membuka GUI
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
# 1. PENGATURAN UTAMA
# ============================================================

DATA_FILE = str(REPO_ROOT / "example" / "synthetic_thickness.xlsx")
SHEET_NAME = 0

X_COL = None
Y_COL = None
Z_COL = "Ketebalan LIM"

# Grid simulasi mengikuti bentuk sebaran data.
GRID_RES = 12.5

# Jika None, spasi bor dihitung otomatis sebagai median nearest-neighbor.
# Jika ingin dipaksa tepat 25 m, ubah menjadi: DRILL_SPACING_OVERRIDE = 25.0
DRILL_SPACING_OVERRIDE = 25.0

# Boundary akhir diperluas 1 x spasi titik bor terluar.
BOUNDARY_BUFFER_MULTIPLIER = 1.0

# Boundary ortogonal seperti batas IUP
BOUNDARY_CELL = 25.0
BOUNDARY_CLOSE_ITERS = 1
FILL_INTERNAL_BOUNDARY_HOLES = True
MASK_OUTSIDE_DATA_SHAPE = True

# Neighborhood sama dengan script estimasi
NMIN = 6
NMAX = 12
ALLOW_FALLBACK_NEAREST = True
NORMALIZED_SEARCH_RADIUS = 1.25

# Variogram NORMAL SCORE - sama untuk Elips dan Cassini
VARIOGRAM_MODEL = "spherical"
NUGGET = 0.06
PARTIAL_SILL = 0.94
SILL = NUGGET + PARTIAL_SILL

# SGS
N_REALIZATIONS = 100
BASE_SEED = 260815

# Banyak kandidat terdekat yang diprekomputasi untuk mempercepat SGS.
# Nilai ini BUKAN jumlah neighbor kriging. Neighbor kriging tetap NMAX=12.
HARD_CANDIDATE_K = 48
GRID_CANDIDATE_K = 128

# Titik grid dianggap berimpit dengan hard data hanya bila jaraknya
# sangat kecil. Jika berimpit, normal-score hard data dihonor persis.
HARD_MATCH_TOL = max(1e-8, GRID_RES * 1e-7)

# Stabilisasi numerik covariance lokal
COV_JITTER_START = 1e-10
COV_JITTER_MAX = 1e-2
VAR_MIN = 1e-10
VAR_MAX = SILL

# Back-transform tails:
# "clip" = nilai simulasi di luar rentang normal-score data dipotong
#          ke min/max ketebalan data. Stabil dan aman untuk ketebalan.
BACKTRANSFORM_TAIL = "clip"

# ============================================================
# OUTPUT BARU - DIPISAH DARI RUN SEBELUMNYA
# ============================================================
# Folder root sengaja berbeda agar hasil run LIM sebelumnya
# tidak tercampur/tertimpa. Di dalamnya output dipisahkan per jenis.
OUTPUT_DIR = REPO_ROOT / "outputs" / "sgs_limonite"

DIR_REALIZATION_CSV = OUTPUT_DIR / "01_realisasi_csv"
DIR_INDIVIDUAL_MAPS = OUTPUT_DIR / "02_peta_individual"
DIR_COMPARE_MAPS = OUTPUT_DIR / "03_peta_perbandingan"
DIR_SUMMARY = OUTPUT_DIR / "04_ringkasan_spasial"
DIR_ENSEMBLE = OUTPUT_DIR / "05_diagnostik_ensemble"
DIR_TABLES = OUTPUT_DIR / "06_tabel_diagnostik"
DIR_ARRAYS = OUTPUT_DIR / "07_array_npy"

SAVE_ALL_REALIZATIONS = True
SAVE_EACH_REALIZATION_CSV = True
SAVE_INDIVIDUAL_REALIZATION_MAPS = True   # 100 Elips + 100 Cassini
SAVE_PAIRED_REALIZATION_MAPS = True       # 100 peta perbandingan 2 model (paired)
SAVE_POSTPROCESS_CSV = True
SAVE_FIGS = True

# Jika proses ekspor gambar terputus, jalankan ulang script yang sama.
# File peta yang sudah ada akan dilewati dan ekspor melanjutkan file yang belum ada.
SKIP_EXISTING_EXPORTS = True

# CSV realisasi disimpan SEGERA setelah setiap realisasi SGS selesai, bukan menunggu
# seluruh peta selesai diekspor. Ini membuat R001-R100 lebih aman terhadap interupsi.
SAVE_REALIZATION_CSV_DURING_SIMULATION = True

# Jangan tampilkan ratusan jendela saat dijalankan dari Anaconda/Jupyter.
SHOW_PLOTS = False
RUN_LOOCV = True

# Contoh realisasi untuk plot ringkasan tambahan (1-based)
EXAMPLE_REALIZATION = 1

# Style peta mengikuti script estimasi yang diberikan pengguna.
N_LEVELS_MAP = 30
POINT_SIZE_SINGLE = 24
POINT_SIZE_COMPARE = 18
COLOR_SCALE_SINGLE = "global"   # semua peta ketebalan memakai skala global yang sama
COLOR_SCALE_COMPARE = "global"  # paired comparison memakai skala warna sama
MAP_DPI = 300

# Ringkasan ensemble 100 realisasi menjadi satu gambar per jenis diagnostik.
# Histogram dan CDF memakai seluruh nilai dari seluruh realisasi.
# Uncertainty plot memakai distribusi 100 nilai pada setiap node grid.
# Variogram ensemble dihitung di ruang normal-score agar dapat dibandingkan
# langsung dengan variogram target SGS (nugget=0.06, partial sill=0.94).
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

# Validasi variogram directional 8 arah.
RUN_DIRECTIONAL_VARIOGRAM_VALIDATION = True
DIRECTIONAL_TOLERANCE_DEG = 11.25   # +/- 11.25 derajat untuk arah berjarak 22.5 derajat
DIRECTIONAL_RANGE_FIT_MIN = max(GRID_RES * 1.5, 20.0)
DIRECTIONAL_RANGE_FIT_MAX_FACTOR = 2.0
DIRECTIONAL_MIN_PAIRS_PER_BIN = 20

MODELS = ["Elips", "Cassini"]
EPS = 1e-12


def prepare_output_folders():
    """Buat struktur folder output baru dan terpisah."""
    folders = [
        OUTPUT_DIR,
        DIR_REALIZATION_CSV,
        DIR_INDIVIDUAL_MAPS / "Elips",
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
    """Kembalikan Path output dan pastikan parent folder tersedia."""
    p = Path(path_like)
    if not p.is_absolute():
        p = OUTPUT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def should_export(path_like):
    """False bila file sudah ada dan mode skip-existing aktif."""
    p = output_file(path_like)
    return not (SKIP_EXISTING_EXPORTS and p.exists())

# Colormap sama dengan script estimasi LIM yang diberikan.
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
# 2. PARAMETER DIRECTIONAL RANGE LIM - HASIL FITTING NLS TERBARU
# ============================================================

# Elips polar NLS
ELLIPSE_MAJOR = 293.99632947
ELLIPSE_MINOR = 163.69907027
ELLIPSE_PHI_AZ = 74.63410107
ELLIPSE_RATIO = ELLIPSE_MAJOR / ELLIPSE_MINOR

# Oval Cassini affine-scaled NLS
CASSINI_A = 165.56486891
CASSINI_C = 228.01740108
CASSINI_SX = 1.01083953
CASSINI_SY = 0.98927671
CASSINI_PHI_AZ = 73.97365979
CASSINI_RATIO_CA = CASSINI_C / CASSINI_A

# Statistik fitting directional range dari script fitting yang diberikan.
DIRECTIONAL_FIT_RMSE = {
    "Elips": 7.74743814,
    "Cassini": 4.52894832,
}

DIRECTIONAL_FIT_R2 = {
    "Elips": 0.97240331,
    "Cassini": 0.99056948,
}


# Delapan arah experimental directional variogram yang dipakai pada fitting NLS.
# Azimuth geologi: 0°=North, 90°=East, clockwise.
# Arah 180°-337.5° adalah pasangan berlawanan dan mempunyai range yang sama.
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

    candidates = glob.glob("*Ketebalan*.xlsx") + glob.glob("*Ketebalan*.xls")

    if len(candidates) > 0:
        print(f"File {file_name} tidak ditemukan. Menggunakan: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Ketebalan.xlsx tidak ditemukan. Letakkan file Excel di folder "
        "yang sama dengan script, atau ubah DATA_FILE."
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
        df, ["Ketebalan LIM", "ketebalan lim", "lim thickness", "ketebalan"]
    )

    if x_col is None or y_col is None or z_col is None:
        print("\nKolom yang tersedia:")
        print(df.columns.tolist())
        raise ValueError(
            "Kolom X, Y, atau Ketebalan LIM tidak terdeteksi. "
            "Isi X_COL, Y_COL, dan Z_COL secara manual."
        )

    data = df[[x_col, y_col, z_col]].copy()
    data.columns = ["X", "Y", "Z"]
    data[["X", "Y"]] = data[["X", "Y"]].apply(pd.to_numeric, errors="coerce")
    data["Z"] = pd.to_numeric(data["Z"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["X", "Y"]).reset_index(drop=True)

    if len(data) < NMIN + 1:
        raise ValueError(
            f"Jumlah data valid hanya {len(data)}. "
            f"Minimal disarankan lebih dari NMIN={NMIN}."
        )

    # Gabungkan koordinat duplikat dengan mean ketebalan agar covariance
    # tidak singular akibat hard data pada posisi yang persis sama.
    n_before = len(data)
    data = data.groupby(["X", "Y"], as_index=False)["Z"].mean()
    n_after = len(data)

    if n_after < n_before:
        print(
            f"Peringatan: {n_before - n_after} data koordinat duplikat "
            "digabung menggunakan mean Ketebalan LIM."
        )

    print("\n" + "=" * 72)
    print("DATA LIM")
    print("=" * 72)
    print(f"File                = {file_path}")
    print(f"Kolom X             = {x_col}")
    print(f"Kolom Y             = {y_col}")
    print(f"Kolom simulasi      = {z_col}")
    print(f"Jumlah hard data    = {len(data)}")
    print(f"Min Ketebalan LIM   = {data['Z'].min():.4f}")
    print(f"Max Ketebalan LIM   = {data['Z'].max():.4f}")
    print(f"Mean Ketebalan LIM  = {data['Z'].mean():.4f}")
    print(f"Std Ketebalan LIM   = {data['Z'].std(ddof=1):.4f}")

    return data


# ============================================================
# 4. NORMAL-SCORE TRANSFORMATION
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

    # Tabel back-transform dibuat dari ordered sample dengan plotting position
    order = np.argsort(z)
    z_sorted = z[order]
    p_sorted = (np.arange(n) + 0.5) / n
    y_sorted = norm.ppf(p_sorted)

    return y, y_sorted, z_sorted


def back_transform(y_values, y_table, z_table, tail_mode="clip"):
    """Back-transform normal score ke unit ketebalan asli."""
    y_values = np.asarray(y_values, dtype=float)

    if tail_mode == "clip":
        return np.interp(
            y_values,
            y_table,
            z_table,
            left=z_table[0],
            right=z_table[-1],
        )

    raise ValueError("BACKTRANSFORM_TAIL saat ini harus 'clip'.")


# ============================================================
# 5. FUNGSI AZIMUTH DAN DIRECTIONAL RANGE
# ============================================================

def azimuth_to_unit_vector(az_deg):
    """
    Azimuth geologi:
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
    Directional range dari affine-scaled Cassinian oval.

    [(X/sx)^2 + (Y/sy)^2 + a^2]^2
      - 4 a^2 (X/sx)^2 - c^4 = 0
    """
    az_deg = np.asarray(az_deg, dtype=float)

    ux, uy = azimuth_to_unit_vector(az_deg)
    theta = phi_az_to_theta_math(CASSINI_PHI_AZ)

    ct = np.cos(theta)
    st = np.sin(theta)

    local_x = ux * ct + uy * st
    local_y = -ux * st + uy * ct

    A = local_x / CASSINI_SX
    B = local_y / CASSINI_SY

    sdir = A**2 + B**2

    qa = sdir**2
    qb = CASSINI_A**2 * (2.0 * sdir - 4.0 * A**2)
    qc = CASSINI_A**4 - CASSINI_C**4

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
    if model_name == "Elips":
        return ellipse_range_from_azimuth(az_deg)
    if model_name == "Cassini":
        return cassini_range_from_azimuth(az_deg)
    raise ValueError("Nama model tidak dikenal.")


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
# 6. VARIOGRAM DAN COVARIANCE NORMAL SCORE
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
    C(h) = sill - gamma(h), dengan C(0) = sill.
    """
    gamma = semivariogram_from_dxdy(dx, dy, model_name)
    cov = SILL - gamma
    return np.maximum(cov, 0.0)


# ============================================================
# 7. GRID SIMULASI - BOUNDARY ORTOGONAL RAPI 25 m
# ============================================================

class OrthogonalBoundary:
    """Boundary ortogonal tanpa Shapely."""
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
    """Estimasi spasi bor tipikal dari median nearest-neighbor distance."""
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
    Boundary dibentuk dari union kotak +/- offset di sekitar titik bor.
    Raster boundary dibuat reguler 25 m, kemudian closing dan fill-holes
    sehingga garis batas horizontal/vertikal, lebih rapi, dan tidak bolong.
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

    # Padding menjaga outer boundary tidak terkikis oleh closing.
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
    """Uji titik terhadap sel boundary; titik tepat di garis tetap diterima."""
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
    """Grid SGS 12.5 m di dalam boundary ortogonal 25 m tanpa lubang."""
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
        raise ValueError("Boundary menghasilkan 0 node simulasi.")

    print("\n" + "=" * 72)
    print("GRID SGS - BOUNDARY ORTOGONAL RAPI")
    print("=" * 72)
    print(f"GRID_RES                  = {GRID_RES:.3f} m")
    print(f"Spasi bor auto (median NN)= {spacing_auto:.3f} m")
    print(f"Spasi bor dipakai         = {drill_spacing:.3f} m")
    print(f"Buffer boundary           = {boundary_buffer:.3f} m")
    print(f"Boundary cell             = {BOUNDARY_CELL:.3f} m")
    print(f"Boundary closing iter     = {BOUNDARY_CLOSE_ITERS}")
    print(f"Fill internal holes       = {FILL_INTERNAL_BOUNDARY_HOLES}")
    print(f"Luas boundary             = {boundary.area:.2f} m2")
    print(f"Jumlah grid rectangle     = {len(grid_points)}")
    print(f"Grid disimulasikan        = {len(valid_points)}")

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
    """Tambahkan garis boundary ortogonal pada peta."""
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
# 8. PRECOMPUTE KANDIDAT NEIGHBOR
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
    Mempercepat SGS dengan precompute kandidat hard data dan grid.
    Urutan akhir kandidat ditentukan oleh normalized anisotropic distance.
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

    # ---------------- GRID NODES ----------------
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

        # Simpan hanya kandidat yang dibutuhkan setelah self diletakkan di akhir.
        keep = min(GRID_CANDIDATE_K, grid_idx.shape[1])
        grid_idx = grid_idx[:, :keep]
        grid_dnorm = grid_dnorm[:, :keep]
    else:
        grid_idx = np.empty((1, 0), dtype=np.int32)
        grid_dnorm = np.empty((1, 0), dtype=float)

    # Exact hard match untuk honoring data bila grid berimpit.
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
      - grid node yang SUDAH disimulasikan pada random path.

    Prioritas mengikuti normalized anisotropic distance.
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

    # Kode tipe: 0 = hard, 1 = simulated grid
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
# 10. SIMPLE KRIGING GAUSSIAN UNTUK SGS
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

    # Coba solve langsung, kemudian tambahkan jitter secara bertahap bila perlu.
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

        # Terima jika numerik masuk akal.
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
# 11. SATU REALISASI SGS
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
        # Honor hard data bila koordinat grid berimpit tepat.
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
# 12. LOOCV DIAGNOSTIK MODEL (GAUSSIAN SIMPLE KRIGING)
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
    """Simpan satu peta SGS dengan style yang sama seperti script estimasi."""
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
        label="Data bor",
        zorder=5,
    )

    plot_orthogonal_boundary(ax, grid_info)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("SGS Ketebalan LIM")

    ax.set_title(
        f"{model_name} - SGS Ketebalan LIM - Realisasi {realization_number:03d}",
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
    """Peta paired realization dua model dengan skala warna yang sama."""
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
            f"Model {model_name} - Realisasi {realization_number:03d}",
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
            label="SGS Ketebalan LIM",
            shrink=0.88,
        )

    fig.suptitle(
        f"Perbandingan SGS Ketebalan LIM - Realisasi {realization_number:03d}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _plot_drill_locations(ax, hard_coords, label="Data bor"):
    """Marker lokasi data bor yang konsisten untuk seluruh peta."""
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
    colorbar_label="Ketebalan LIM",
    fixed_vmin=None,
    fixed_vmax=None,
):
    """Perbandingan tiga peta dengan marker data bor pada seluruh panel.

    Bila fixed_vmin/fixed_vmax diberikan, seluruh panel memakai rentang warna
    yang persis sama. Ini dipakai untuk seluruh peta dalam satuan ketebalan.
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
    """Difference map E-type Cassini - Elips."""
    XX = grid_info["XX"]
    YY = grid_info["YY"]
    diff_map = etype_maps["Cassini"] - etype_maps["Elips"]
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
    ax.set_title("Cassini - Elips", fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=7)
    _add_north_arrow(ax)
    fig.colorbar(cf, ax=ax, shrink=0.86, label="Perbedaan E-type")
    fig.suptitle("Perbedaan E-type SGS LIM: Cassini - Elips", fontsize=13, fontweight="bold")

    if SAVE_FIGS:
        fig.savefig(output_file(file_name), dpi=MAP_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def build_variogram_pair_sample(valid_points):
    """Sampel pasangan grid bersama untuk variogram ensemble.

    Pasangan terdiri atas kombinasi lokal dan global. Lag memakai jarak Euclidean
    fisik, sedangkan arah memakai azimuth geologi axial 0-180 derajat.
    """
    points = np.asarray(valid_points, dtype=float)
    n = len(points)
    if n < 2:
        raise ValueError("Minimal dua node grid diperlukan untuk variogram ensemble.")

    rng = np.random.default_rng(BASE_SEED + 99173)
    pair_blocks = []

    # Pasangan lokal untuk menjaga representasi lag kecil.
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

    # Pasangan global untuk lag sedang-besar.
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
    print(f"Jumlah pasangan dipakai = {len(pairs)}")
    print(f"Lag maksimum            = {max_lag:.3f} m")
    print(f"Jumlah lag bin          = {VARIOGRAM_N_LAGS}")

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
    """Variogram experimental satu realisasi pada lag bin bersama."""
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
    """Rata-rata angular variogram teoritis pada lag fisik (meter)."""
    az = np.linspace(0.0, 180.0, 181, endpoint=False)
    az_rad = np.deg2rad(az)
    out = np.empty(len(lag_centers), dtype=float)
    for i, h in enumerate(lag_centers):
        dx = h * np.sin(az_rad)
        dy = h * np.cos(az_rad)
        out[i] = float(np.mean(semivariogram_from_dxdy(dx, dy, model_name)))
    return out



def axial_angular_difference(azimuth_deg, target_deg):
    """Selisih sudut axial 0-180; arah d dan d+180 dianggap sama."""
    azimuth_deg = np.asarray(azimuth_deg, dtype=float) % 180.0
    target_deg = float(target_deg) % 180.0
    return np.abs(((azimuth_deg - target_deg + 90.0) % 180.0) - 90.0)


def build_directional_pair_infos(pair_info):
    """Turunkan delapan subset pasangan dari sampel pasangan grid bersama."""
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
    """Experimental variogram satu realisasi pada satu arah."""
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
    Recalculate directional experimental variogram dari hard data menggunakan
    lag bins dan tolerance VALIDASI. Ini adalah pemeriksaan ulang independen;
    nilainya hanya identik dengan experimental variogram fitting asal jika
    lag width/tolerance asal memang sama dengan setting validasi ini.
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
    """Spherical variogram fixed nugget/sill untuk range tertentu."""
    lags = np.asarray(lags, dtype=float)
    r = max(float(range_m), EPS)
    t = lags / r
    gamma = NUGGET + PARTIAL_SILL * spherical_core(t)
    return np.minimum(gamma, SILL)


def fit_directional_range_fixed_nugget_sill(lags, gamma, counts=None, max_lag=None):
    """
    Fit hanya parameter range; nugget dan sill tetap sama dengan model SGS.
    Digunakan untuk melihat apakah range pada 100 realisasi dapat mereproduksi
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
    """Tabel validasi directional berdasarkan 100 realisasi."""
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
    """Simpan kurva ensemble directional lengkap untuk audit."""
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
            DIR_ENSEMBLE / f"variogram_directional_8arah_{model_name}.csv",
            index=False,
        )



def plot_recovered_directional_ranges(
    directional_validation_df,
    file_name="18_recovered_directional_range_8arah_2model.png",
):
    """Validasi langsung directional range 8 arah untuk dua model."""
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
            label="Range 8-arah input"
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
            label="Median range simulasi + 95% interval",
        )

        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Azimuth (°)")
        ax.set_xticks(DIRECTIONAL_AZIMUTHS)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Directional range (m)")
    fig.suptitle(
        "Validasi Reproduksi Directional Range 8 Arah dari 100 Realisasi SGS",
        fontsize=13, fontweight="bold"
    )

    if SAVE_FIGS:
        fig.savefig(
            output_file(Path("05_diagnostik_ensemble") / file_name),
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
    Satu gambar 2x4 untuk tiap model.
    Setiap panel arah menampilkan 100 garis variogram realisasi agar pola
    reproduksi antar realisasi terlihat jelas. Tidak ada garis median maupun
    garis target model, karena setiap arah memang mempunyai range yang berbeda.
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
                    label="100 realisasi" if r == 0 else None,
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
        axes[0].set_ylabel("Semivariance normal-score")
        axes[4].set_ylabel("Semivariance normal-score")
        axes[0].legend(fontsize=7, loc="best")

        fig.suptitle(
            f"Variogram Directional 8 Arah - 100 Realisasi SGS {model_name}",
            fontsize=14, fontweight="bold"
        )
        if SAVE_FIGS:
            fig.savefig(
                DIR_ENSEMBLE /
                f"17_variogram_directional_8arah_validasi_{model_name}.png",
                dpi=MAP_DPI,
                bbox_inches="tight",
            )

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def histogram_validation_table(realizations, z_original):
    """Ringkasan reproduksi distribusi global sebagai alat validasi histogram."""
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

        # Quantile RMSE memberi ringkasan sederhana kemiripan histogram tanpa
        # memakai p-value yang tidak tepat untuk data spasial yang saling berkorelasi.
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
    Satu gambar per model. Setiap gambar menampilkan 100 histogram garis
    (step histogram) agar bentuknya tetap mengikuti histogram asli dan tidak
    terlalu smooth. Tidak ada garis rata-rata.
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
                label="100 realisasi" if r == 0 else None,
            )

        ax.step(
            centers, raw_rel, where="mid",
            linewidth=2.0, linestyle="--",
            label="Histogram data bor",
        )

        ax.set_title(
            f"Histogram 100 Realisasi SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Ketebalan LIM")
        ax.set_ylabel("Frekuensi relatif")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_diagnostik_ensemble") /
                    f"13_histogram_100_realisasi_{model_name.lower()}.png"
                ),
                dpi=MAP_DPI,
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def plot_uncertainty_ensemble(post, file_name="14_uncertainty_plot_100_realisasi_2model.png"):
    """Satu gambar uncertainty plot dari 100 realisasi untuk dua model."""
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
        ax.set_xlabel("Node grid (urutan E-type referensi)")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Ketebalan LIM")
    fig.suptitle("Uncertainty plot 100 realisasi SGS - interval simulasi per node", fontsize=13, fontweight="bold")

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
    """Satu gambar CDF per model dengan 100 kurva realisasi terlihat."""
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
                label="100 realisasi" if r == 0 else None,
            )

        curves = np.asarray(curves, dtype=float)
        mean_cdf = np.nanmean(curves, axis=0)
        ax.plot(x, mean_cdf, linewidth=2.4, label="Rata-rata CDF 100 realisasi")
        ax.plot(x, raw_cdf, linewidth=2.0, linestyle="--", label="CDF data bor")

        ax.set_title(
            f"CDF 100 Realisasi SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Ketebalan LIM")
        ax.set_ylabel("Probabilitas kumulatif")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_diagnostik_ensemble") /
                    f"15_CDF_100_realisasi_{model_name.lower()}.png"
                ),
                dpi=MAP_DPI,
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


def plot_variogram_100_realizations_by_model(variogram_curves, pair_info):
    """Satu gambar variogram per model dengan 100 garis realisasi."""
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
                label="100 variogram realisasi" if r == 0 else None,
            )

        ax.plot(
            centers, mean_curve,
            linewidth=2.4, marker="o", markersize=3,
            label="Mean 100 realisasi",
        )
        ax.plot(
            centers, median_curve,
            linewidth=1.8, linestyle="--",
            label="Median 100 realisasi",
        )
        ax.plot(
            centers, target,
            linewidth=2.2, linestyle=":",
            label="Variogram target",
        )
        ax.axhline(SILL, linewidth=1.0, linestyle="-.", label="Sill")

        ax.set_title(
            f"Variogram 100 Realisasi SGS - {model_name}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Semivariance normal-score")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9)

        if SAVE_FIGS:
            fig.savefig(
                output_file(
                    Path("05_diagnostik_ensemble") /
                    f"16_variogram_100_realisasi_{model_name.lower()}.png"
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
    print("FOLDER OUTPUT RUN BARU")
    print("=" * 72)
    print(f"Root output            = {OUTPUT_DIR.resolve()}")
    print(f"CSV realisasi          = {DIR_REALIZATION_CSV.resolve()}")
    print(f"Peta individual        = {DIR_INDIVIDUAL_MAPS.resolve()}")
    print(f"Peta perbandingan      = {DIR_COMPARE_MAPS.resolve()}")
    print(f"Ringkasan/diagnostik   = {DIR_SUMMARY.resolve()}")

    data = load_thickness_data()
    hard_coords = data[["X", "Y"]].values.astype(float)
    z_original = data["Z"].values.astype(float)

    # Normal-score transform
    hard_gauss, y_table, z_table = normal_score_transform(z_original)

    print("\n" + "=" * 72)
    print("NORMAL SCORE")
    print("=" * 72)
    print(f"Mean normal score     = {np.mean(hard_gauss):.6f}")
    print(f"Std normal score      = {np.std(hard_gauss, ddof=1):.6f}")
    print(f"NUGGET                = {NUGGET:.4f}")
    print(f"PARTIAL_SILL          = {PARTIAL_SILL:.4f}")
    print(f"TOTAL SILL            = {SILL:.4f}")

    print("\n" + "=" * 72)
    print("PARAMETER DIRECTIONAL RANGE")
    print("=" * 72)
    print(
        f"Elips     : Az={ELLIPSE_PHI_AZ:.5f}°, "
        f"A={ELLIPSE_MAJOR:.5f} m, B={ELLIPSE_MINOR:.5f} m, "
        f"A/B={ELLIPSE_RATIO:.5f}"
    )
    print(
        f"Cassini   : Az={CASSINI_PHI_AZ:.5f}°, "
        f"a={CASSINI_A:.5f} m, c={CASSINI_C:.5f} m, "
        f"c/a={CASSINI_RATIO_CA:.5f}, "
        f"sx={CASSINI_SX:.5f}, sy={CASSINI_SY:.5f}"
    )
    print(f"NMIN={NMIN}, NMAX={NMAX}, search h/R={NORMALIZED_SEARCH_RADIUS}")
    print(f"Jumlah realisasi      = {N_REALIZATIONS}")
    print(f"Base seed             = {BASE_SEED}")

    grid_info = build_grid(data)
    valid_points = grid_info["valid_points"]
    n_valid = len(valid_points)

    # Candidate cache per model.
    caches = {
        model_name: build_candidate_cache(valid_points, hard_coords, model_name)
        for model_name in MODELS
    }

    # Optional LOOCV sebelum simulasi.
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

    # Simpan realisasi pada unit asli dalam float32 untuk mengurangi RAM.
    realizations = {
        model_name: np.empty((N_REALIZATIONS, n_valid), dtype=np.float32)
        for model_name in MODELS
    }

    # Variogram ensemble dihitung langsung dari normal-score tiap realisasi.
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
    print(f"MENJALANKAN SGS {N_REALIZATIONS} REALISASI - 2 MODEL")
    print("=" * 72)
    print(
        "Setiap pasangan realisasi Elips dan Cassini menggunakan "
        "random path dan Gaussian innovation yang sama."
    )

    for r in range(N_REALIZATIONS):
        realization_number = r + 1
        seed = BASE_SEED + r
        rng = np.random.default_rng(seed)

        # PAIRED path dan innovations untuk kedua model.
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

        # Simpan paired CSV segera setelah kedua model pada realisasi ini selesai.
        # Jadi hasil R001-R100 tidak bergantung pada proses ekspor ratusan peta setelah SGS.
        if SAVE_EACH_REALIZATION_CSV and SAVE_REALIZATION_CSV_DURING_SIMULATION:
            csv_path = DIR_REALIZATION_CSV / (
                f"R{realization_number:03d}_SGS_LIM_Elips_Cassini.csv"
            )
            pd.DataFrame({
                "X": valid_points[:, 0],
                "Y": valid_points[:, 1],
                **{f"SGS_{m}": realizations[m][r] for m in MODELS},
            }).to_csv(csv_path, index=False)

        # Checkpoint diagnostik ringan setiap 5 realisasi.
        if realization_number % 5 == 0 or realization_number == N_REALIZATIONS:
            pd.DataFrame(diagnostics_rows).to_csv(
                DIR_TABLES / "diagnostic_checkpoint.csv", index=False
            )

        dt = time.time() - t0
        print(
            f"Realisasi {realization_number:03d}/{N_REALIZATIONS} selesai "
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
    diagnostics_df.to_csv(DIR_TABLES / "diagnostic_100_realisasi_2model.csv", index=False)
    comparison_df.to_csv(DIR_TABLES / "comparison_summary_Elips_Cassini.csv", index=False)

    if RUN_LOOCV:
        pd.DataFrame(cv_summary_rows).to_csv(DIR_TABLES / "LOOCV_summary.csv", index=False)
        for model_name in MODELS:
            cv_details[model_name].to_csv(
                DIR_TABLES / f"LOOCV_detail_{model_name}.csv", index=False
            )

    if SAVE_ALL_REALIZATIONS:
        for model_name in MODELS:
            np.save(
                DIR_ARRAYS / f"SGS_LIM_{model_name}_{N_REALIZATIONS}realisasi.npy",
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
        "Ketebalan_LIM_sorted": z_table,
    }).to_csv(DIR_TABLES / "normal_score_backtransform_table.csv", index=False)

    # Tabel validasi distribusi global.
    histogram_validation_df = histogram_validation_table(
        realizations, z_original
    )
    histogram_validation_df.to_csv(
        DIR_ENSEMBLE / "histogram_validation_summary.csv",
        index=False,
    )

    if RUN_ENSEMBLE_DIAGNOSTICS:
        # Simpan variogram omnidirectional 100 realisasi.
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
            DIR_ENSEMBLE / "variogram_ensemble_100_realisasi.csv",
            index=False,
        )

        if RUN_DIRECTIONAL_VARIOGRAM_VALIDATION:
            DIRECTIONAL_VALIDATION_TABLE.to_csv(
                DIR_TABLES / "directional_range_8arah_input.csv",
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

    # Skala global sama untuk SELURUH peta yang bersatuan ketebalan LIM.
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
        f"Skala global peta ketebalan = "
        f"{global_vmin:.4f} s.d. {global_vmax:.4f}"
    )

    print("\n" + "=" * 72)
    print("MENYIMPAN 100 REALISASI DAN PETA SGS 2 MODEL")
    print("=" * 72)
    print(f"Root folder output = {OUTPUT_DIR.resolve()}")

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
                DIR_REALIZATION_CSV / f"R{real_no:03d}_SGS_LIM_Elips_Cassini.csv",
                index=False,
            )

        if SAVE_INDIVIDUAL_REALIZATION_MAPS:
            for model_name in MODELS:
                rel_file = Path("02_peta_individual") / model_name / (
                    f"R{real_no:03d}_Peta_SGS_LIM_{model_name}.png"
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
            rel_compare = Path("03_peta_perbandingan") / (
                f"R{real_no:03d}_Peta_Perbandingan_2Model.png"
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

        # Bersihkan object figure/memori secara berkala untuk ekspor ratusan peta.
        plt.close("all")
        if real_no % 5 == 0:
            gc.collect()

        if real_no % 10 == 0 or real_no == 1:
            print(f"  ekspor peta realisasi {real_no:03d}/{N_REALIZATIONS} selesai/terverifikasi")

    # ---------------- PLOTS RINGKASAN SPASIAL ----------------
    example_idx = int(np.clip(EXAMPLE_REALIZATION - 1, 0, N_REALIZATIONS - 1))
    example_maps = {
        m: valid_to_full_grid(realizations[m][example_idx], grid_info)
        for m in MODELS
    }
    plot_model_maps(
        example_maps,
        grid_info,
        hard_coords,
        {m: f"{m} - Realisasi {example_idx + 1}" for m in MODELS},
        "SGS Ketebalan LIM - Contoh Paired Realization 2 Model",
        Path("04_ringkasan_spasial") / "01_contoh_realisasi_2model.png",
        "Ketebalan LIM",
        fixed_vmin=global_vmin,
        fixed_vmax=global_vmax,
    )

    summary_specs = [
        ("Etype", "E-type", "E-type dari 100 Realisasi SGS - Ketebalan LIM", "02_Etype_2model.png", "Ketebalan LIM"),
        ("SD", "SD", "Standard Deviation 100 Realisasi SGS - Ketebalan LIM", "03_SD_2model.png", "SD Ketebalan LIM"),
        ("Variance", "Variance", "Variance 100 Realisasi SGS - Ketebalan LIM", "04_Variance_2model.png", "Variance"),
        ("CL95_Lower", "P2.5", "Lower 95% Simulation Interval SGS - Ketebalan LIM", "05_P025_2model.png", "Ketebalan LIM"),
        ("CL95_Upper", "P97.5", "Upper 95% Simulation Interval SGS - Ketebalan LIM", "06_P975_2model.png", "Ketebalan LIM"),
        ("CL95_Width", "Width P2.5-P97.5", "Lebar 95% Simulation Interval SGS - Ketebalan LIM", "07_interval95_width_2model.png", "Lebar interval"),
        ("P25", "P25", "P25 dari 100 Realisasi SGS - Ketebalan LIM", "08_P25_2model.png", "Ketebalan LIM"),
        ("P50", "P50", "P50 dari 100 Realisasi SGS - Ketebalan LIM", "09_P50_2model.png", "Ketebalan LIM"),
        ("P75", "P75", "P75 dari 100 Realisasi SGS - Ketebalan LIM", "10_P75_2model.png", "Ketebalan LIM"),
        ("P95", "P95", "P95 dari 100 Realisasi SGS - Ketebalan LIM", "11_P95_2model.png", "Ketebalan LIM"),
    ]

    # Output yang memang bersatuan ketebalan memakai skala GLOBAL yang sama.
    # SD, Variance, dan Width punya satuan berbeda sehingga tidak dipaksa memakai
    # skala ketebalan; namun skala antar model tetap identik di dalam tiap gambar.
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
            Path("04_ringkasan_spasial") / filename,
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
            Path("04_ringkasan_spasial") / "12_difference_Etype_2model.png",
        )

    # Histogram mean/std per realisasi (tetap dipertahankan dari script asal).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model_name in MODELS:
        sub = diagnostics_df[diagnostics_df["Model"] == model_name]
        axes[0].hist(sub["Sim_mean"], bins=15, alpha=0.45, label=model_name)
        axes[1].hist(sub["Sim_std"], bins=15, alpha=0.45, label=model_name)
    axes[0].axvline(np.mean(z_original), linestyle="--", linewidth=1.5, label="Hard data mean")
    axes[1].axvline(np.std(z_original, ddof=1), linestyle="--", linewidth=1.5, label="Hard data std")
    axes[0].set_title("Mean per Realisasi")
    axes[1].set_title("Standard Deviation per Realisasi")
    axes[0].set_xlabel("Mean Ketebalan LIM")
    axes[1].set_xlabel("Std Ketebalan LIM")
    axes[0].set_ylabel("Frekuensi")
    axes[1].set_ylabel("Frekuensi")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    if SAVE_FIGS:
        fig.savefig(DIR_ENSEMBLE / "12b_histogram_mean_std_per_realisasi.png", dpi=250, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    # ---------------- OUTPUT DIAGNOSTIK ENSEMBLE ----------------
    if RUN_ENSEMBLE_DIAGNOSTICS:
        plot_histogram_100_realizations_by_model(
            realizations,
            z_original,
        )
        plot_uncertainty_ensemble(
            post,
            Path("05_diagnostik_ensemble") /
            "14_uncertainty_plot_100_realisasi_2model.png",
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
    print("SGS SELESAI")
    print("=" * 72)
    print(comparison_df.round(5).to_string(index=False))
    print(f"\nWaktu total = {elapsed / 60.0:.2f} menit")
    print(f"Output tersimpan di root baru: {OUTPUT_DIR.resolve()}")
    print("\nStruktur folder output:")
    print(f"- 01_realisasi_csv      : {DIR_REALIZATION_CSV}")
    print(f"- 02_peta_individual    : {DIR_INDIVIDUAL_MAPS}")
    print(f"- 03_peta_perbandingan  : {DIR_COMPARE_MAPS}")
    print(f"- 04_ringkasan_spasial  : {DIR_SUMMARY}")
    print(f"- 05_diagnostik_ensemble: {DIR_ENSEMBLE}")
    print(f"- 06_tabel_diagnostik   : {DIR_TABLES}")
    print(f"- 07_array_npy          : {DIR_ARRAYS}")
    print("\nOutput utama:")
    print(f"1. {N_REALIZATIONS} CSV paired 2 model (R001 ... R{N_REALIZATIONS:03d})")
    print(f"2. {N_REALIZATIONS} peta SGS Elips")
    print(f"3. {N_REALIZATIONS} peta SGS Cassini")
    print(f"4. {N_REALIZATIONS} peta perbandingan paired 2 model")
    print("5. E-type, SD, variance, interval simulasi, percentile, dan difference map Cassini-Elips")
    print("6. Histogram: 100 garis/step realisasi per model (Y = Frekuensi relatif)")
    print("7. Uncertainty plot 100 realisasi")
    print("8. CDF: 100 garis realisasi dalam 1 gambar untuk tiap model")
    print("9. Variogram: 100 garis realisasi dalam 1 gambar untuk tiap model")
    print("10. Variogram directional 8 arah: 2 gambar (Elips, Cassini)")
    print("11. Tabel recovered directional range dan RMSE variogram")
    print("12. Plot recovered directional range 8 arah (input vs model vs simulasi)")

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