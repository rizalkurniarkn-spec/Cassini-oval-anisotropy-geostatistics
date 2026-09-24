# ============================================================
# ORDINARY KRIGING OF LIMONITE THICKNESS - ELLIPSE AND CASSINI OVAL
# Models: 1) Ellipse | 2) Affine-scaled Cassini Oval
#
# Public-repository version. The default input is a synthetic example dataset.
# Replace DATA_FILE with an authorized local dataset to reproduce private-data runs.
#
# Revisi utama:
# - Boundary ortogonal mengikuti sebaran data, offset 25 m (tanpa Shapely)
# - Lubang internal boundary diisi
# - Nilai Ketebalan LIM kosong pada Excel -> 0
# - Search radius SEMUA model = 400 m, berbentuk LINGKARAN
# - Anisotropi hanya memengaruhi variogram/kriging, bukan search radius
# - Ordinary Kriging + spherical variogram
# ============================================================

import os, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_fill_holes, binary_closing

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
DATA_FILE, SHEET_NAME = str(REPO_ROOT / "example" / "synthetic_thickness.xlsx"), 0
X_COL, Y_COL, Z_COL = None, None, "Ketebalan LIM"
GRID_RES = 12.5
DRILL_SPACING = 25.0
BOUNDARY_OFFSET = DRILL_SPACING
FILL_INTERNAL_BOUNDARY_HOLES = True
MASK_OUTSIDE_BOUNDARY = True
BOUNDARY_CELL = DRILL_SPACING        # raster boundary 25 m agar bentuk lebih rapi
BOUNDARY_CLOSE_ITERS = 1             # tutup celah/lekukan kecil pada boundary

# Search neighborhood sama untuk SEMUA model
SEARCH_RADIUS = 400.0            # meter
NMIN, NMAX = 6, 12
ALLOW_FEWER_WITHIN_RADIUS = True # bila kandidat < NMIN, tetap pakai kandidat yang tersedia
MIN_FALLBACK_NEIGHBORS = 1       # agar grid di dalam boundary tidak bolong, tetap hanya memakai data <= 400 m

NUGGET, PARTIAL_SILL = 5.76, 69.61
SILL = NUGGET + PARTIAL_SILL
EPS = 1e-10
SAVE_FIGS, EXPORT_EXCEL = True, True

# Semua hasil otomatis disimpan ke folder baru ini
OUTPUT_DIR = str(REPO_ROOT / "outputs" / "ordinary_kriging_limonite")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_NEIGHBOR_DIAGNOSTIC = False
DIAGNOSTIC_TARGET_MODE = "sample_index"  # sample_index | coordinate
DIAGNOSTIC_SAMPLE_INDEX = 0
DIAGNOSTIC_X = DIAGNOSTIC_Y = None
DIAGNOSTIC_SHOW_LABELS = True
DIAGNOSTIC_ZOOM_FACTOR = 1.12

# ============================================================
# 2. PARAMETER NLS MODEL DIRECTIONAL RANGE
# ============================================================
ELLIPSE_MAJOR, ELLIPSE_MINOR, ELLIPSE_PHI_AZ = 313.78909, 162.08402, 75.82277
CASSINI_A, CASSINI_C = 176.84564, 235.81139
CASSINI_SX, CASSINI_SY, CASSINI_PHI_AZ = 1.02531, 0.97532, 75.50630
MODELS = ["Elips", "Cassini"]

print("=" * 70, "\nPARAMETER ESTIMASI\n" + "=" * 70, sep="")
print(f"GRID_RES              = {GRID_RES} m")
print(f"BOUNDARY_OFFSET       = {BOUNDARY_OFFSET} m")
print(f"SEARCH_RADIUS         = {SEARCH_RADIUS} m (lingkaran, semua model)")
print(f"NMIN / NMAX           = {NMIN} / {NMAX}")
print(f"NUGGET / PARTIAL SILL = {NUGGET} / {PARTIAL_SILL}")
print("Anisotropi digunakan pada variogram, bukan pada search neighborhood.")
print(f"Folder output          = {os.path.abspath(OUTPUT_DIR)}\n")

# ============================================================
# 3. BACA DATA EXCEL
# ============================================================
def find_excel_file(name):
    if os.path.exists(name): return name
    c = glob.glob("*Ketebalan*.xlsx") + glob.glob("*Ketebalan*.xls")
    if c: return c[0]
    raise FileNotFoundError("Ketebalan.xlsx tidak ditemukan.")


def detect_column(df, candidates):
    cols = list(df.columns); low = {str(c).strip().lower(): c for c in cols}
    for key in candidates:
        if key.lower() in low: return low[key.lower()]
    for c in cols:
        cl = str(c).strip().lower()
        if any(k.lower() in cl for k in candidates): return c
    return None


def load_thickness_data():
    path = find_excel_file(DATA_FILE)
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    xc = X_COL or detect_column(df, ["x", "easting", "east"])
    yc = Y_COL or detect_column(df, ["y", "northing", "north"])
    zc = Z_COL if Z_COL in df.columns else detect_column(df, ["ketebalan lim", "ketebalan", "z"])
    if xc is None or yc is None or zc is None:
        raise ValueError(f"Kolom X/Y/Z tidak terdeteksi. Kolom tersedia: {df.columns.tolist()}")

    d = df[[xc, yc, zc]].copy(); d.columns = ["X", "Y", "Z"]
    d[["X", "Y"]] = d[["X", "Y"]].apply(pd.to_numeric, errors="coerce")
    d["Z"] = pd.to_numeric(d["Z"], errors="coerce").fillna(0.0)  # kosong = 0
    d = d.dropna(subset=["X", "Y"]).reset_index(drop=True)

    print("=" * 70, "\nDATA BERHASIL DIBACA\n" + "=" * 70, sep="")
    print(f"File = {path} | X = {xc} | Y = {yc} | Z = {zc}")
    print(f"Jumlah data = {len(d)} | Z min/max/mean = {d.Z.min():.3f} / {d.Z.max():.3f} / {d.Z.mean():.3f}\n")
    return d


data = load_thickness_data()
coords = data[["X", "Y"]].values
x_data, y_data, z_data = data.X.values, data.Y.values, data.Z.values

# ============================================================
# 4. FUNGSI AZIMUTH GEOLOGI
# ============================================================
def azimuth_to_unit_vector(az):
    a = np.deg2rad(np.asarray(az, float)); return np.sin(a), np.cos(a)


def pair_azimuth_from_dxdy(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def phi_az_to_theta_math(phi):
    return np.deg2rad((90.0 - phi) % 360.0)

# ============================================================
# 5. FUNGSI RANGE MASING-MASING MODEL
# ============================================================


def ellipse_range_from_azimuth(az):
    ux, uy = azimuth_to_unit_vector(az)
    p = np.deg2rad(ELLIPSE_PHI_AZ)
    major = ux*np.sin(p) + uy*np.cos(p)
    minor = ux*np.sin(p + np.pi/2) + uy*np.cos(p + np.pi/2)
    return 1.0 / np.sqrt((major/ELLIPSE_MAJOR)**2 + (minor/ELLIPSE_MINOR)**2)


def cassini_range_from_azimuth(az):
    az = np.asarray(az, float); ux, uy = azimuth_to_unit_vector(az)
    t = phi_az_to_theta_math(CASSINI_PHI_AZ); ct, st = np.cos(t), np.sin(t)
    X = (ux*ct + uy*st) / CASSINI_SX
    Y = (-ux*st + uy*ct) / CASSINI_SY
    S = X**2 + Y**2
    qa = S**2
    qb = CASSINI_A**2 * (2*S - 4*X**2)
    qc = CASSINI_A**4 - CASSINI_C**4
    disc = np.maximum(qb**2 - 4*qa*qc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = (-qb + np.sqrt(disc)) / (2*qa)
        q2 = (-qb - np.sqrt(disc)) / (2*qa)
    q = np.nanmax(np.where(np.stack([q1, q2]) > 0, np.stack([q1, q2]), np.nan), axis=0)
    return np.sqrt(q)


def get_directional_range(az, model):
    return {"Elips": ellipse_range_from_azimuth,
            "Cassini": cassini_range_from_azimuth}[model](az)

# ============================================================
# 6. VARIOGRAM SPHERICAL
# ============================================================
def spherical_variogram_from_dxdy(dx, dy, model):
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    h = np.hypot(dx, dy)
    a = get_directional_range(pair_azimuth_from_dxdy(dx, dy), model)
    if np.any(~np.isfinite(a) | (a <= EPS)):
        raise ValueError(f"Directional range tidak valid: {model}")
    t = h / a
    return np.where(h <= EPS, 0.0,
                    np.where(t <= 1.0, NUGGET + PARTIAL_SILL*(1.5*t - 0.5*t**3), SILL))

# ============================================================
# 7. SEARCH RADIUS LINGKARAN 400 m + ORDINARY KRIGING
# ============================================================
def get_neighbors(x0, y0, coords_train):
    """Semua model: Euclidean search circle <= 400 m, maksimum NMAX."""
    dist = np.hypot(coords_train[:, 0] - x0, coords_train[:, 1] - y0)
    idx = np.where(dist <= SEARCH_RADIUS + EPS)[0]
    if len(idx) >= NMIN:
        return idx[np.argsort(dist[idx])[:NMAX]]
    if ALLOW_FEWER_WITHIN_RADIUS and len(idx) >= MIN_FALLBACK_NEIGHBORS:
        return idx[np.argsort(dist[idx])[:NMAX]]
    return None


def ordinary_kriging_point(x0, y0, coords_train, z_train, model):
    idx = get_neighbors(x0, y0, coords_train)
    if idx is None: return np.nan, np.nan, 0, np.nan, np.nan, np.nan
    xy, z = coords_train[idx], z_train[idx]; n = len(idx)
    dx = xy[None, :, 0] - xy[:, None, 0]
    dy = xy[None, :, 1] - xy[:, None, 1]
    G = spherical_variogram_from_dxdy(dx, dy, model); np.fill_diagonal(G, 0.0)
    K = np.zeros((n+1, n+1)); K[:n, :n] = G; K[:n, n] = K[n, :n] = 1.0
    g0 = spherical_variogram_from_dxdy(xy[:, 0]-x0, xy[:, 1]-y0, model)
    rhs = np.r_[g0, 1.0]
    try: sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError: sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    w, mu = sol[:n], sol[n]
    pred = float(w @ z); kvar = float(w @ g0 + mu)
    kvar = np.nan if not np.isfinite(kvar) else max(kvar, 0.0)
    return pred, kvar, n, int(np.sum(w < 0)), float(w.min()), float(w.max())

# ============================================================
# 8. DIAGNOSTIK SEARCH NEIGHBORHOOD DAN BOBOT KRIGING
# ============================================================
def diagnostic_target():
    if DIAGNOSTIC_TARGET_MODE.lower() == "sample_index":
        i = int(DIAGNOSTIC_SAMPLE_INDEX)
        mask = np.ones(len(data), bool); mask[i] = False
        return data.X[i], data.Y[i], data.Z[i], coords[mask], z_data[mask], np.arange(len(data))[mask]
    x0 = float(DIAGNOSTIC_X if DIAGNOSTIC_X is not None else data.X.mean())
    y0 = float(DIAGNOSTIC_Y if DIAGNOSTIC_Y is not None else data.Y.mean())
    return x0, y0, np.nan, coords, z_data, np.arange(len(data))


def solve_diagnostic(x0, y0, xytrain, ztrain, model, ids):
    dist = np.hypot(xytrain[:,0]-x0, xytrain[:,1]-y0)
    idx = get_neighbors(x0, y0, xytrain)
    if idx is None: return None
    xy, z = xytrain[idx], ztrain[idx]; n = len(idx)
    dx = xy[None,:,0]-xy[:,None,0]; dy = xy[None,:,1]-xy[:,None,1]
    G = spherical_variogram_from_dxdy(dx, dy, model); np.fill_diagonal(G, 0)
    K = np.zeros((n+1,n+1)); K[:n,:n]=G; K[:n,n]=K[n,:n]=1
    g0 = spherical_variogram_from_dxdy(xy[:,0]-x0, xy[:,1]-y0, model)
    try: sol = np.linalg.solve(K, np.r_[g0,1.])
    except np.linalg.LinAlgError: sol = np.linalg.lstsq(K, np.r_[g0,1.], rcond=None)[0]
    w, mu = sol[:n], sol[n]
    az = pair_azimuth_from_dxdy(xy[:,0]-x0, xy[:,1]-y0)
    R = get_directional_range(az, model)
    tab = pd.DataFrame({"N":[f"N{i+1}" for i in range(n)], "ID":ids[idx]+1, "Z":z,
                        "h":dist[idx], "Azimuth":az, "R":R, "Gamma":g0,
                        "Lambda":w, "Lambda_Z":w*z})
    return dict(model=model, x0=x0, y0=y0, train=xytrain, idx=idx, table=tab,
                pred=float(w@z), var=max(float(w@g0+mu),0), w=w,
                n_inside=int(np.sum(dist <= SEARCH_RADIUS+EPS)), cond=float(np.linalg.cond(K)))


def plot_diagnostic(d, observed=np.nan):
    fig = plt.figure(figsize=(18,7)); gs = fig.add_gridspec(1,3,width_ratios=[1.2,.48,1.15])
    ax, info, at = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])
    info.axis("off"); at.axis("off")
    x0,y0=d["x0"],d["y0"]; tr=d["train"]; idx=d["idx"]; w=d["w"]
    ax.scatter(tr[:,0], tr[:,1], s=12, alpha=.25, label="Conditioning data")
    ax.add_patch(Circle((x0,y0), SEARCH_RADIUS, fill=False, ls="--", lw=2,
                        label=f"Search circle = {SEARCH_RADIUS:.0f} m"))
    # anisotropic variogram range shown only as reference
    az=np.linspace(0,360,720); rr=get_directional_range(az,d["model"]); ar=np.deg2rad(az)
    ax.plot(x0+rr*np.sin(ar), y0+rr*np.cos(ar), lw=1.6, label=f"{d['model']} variogram range")
    pos,neg=idx[w>=0],idx[w<0]
    if len(pos): ax.scatter(tr[pos,0],tr[pos,1],s=55,edgecolor="k",label="Selected λ ≥ 0")
    if len(neg): ax.scatter(tr[neg,0],tr[neg,1],s=65,marker="X",edgecolor="k",label="Selected λ < 0")
    ax.scatter([x0],[y0],s=170,marker="*",edgecolor="k",zorder=8,label="Target")
    if DIAGNOSTIC_SHOW_LABELS:
        for j,k in enumerate(idx): ax.annotate(f"N{j+1}",(tr[k,0],tr[k,1]),xytext=(4,4),textcoords="offset points",fontsize=7)
    r=SEARCH_RADIUS*DIAGNOSTIC_ZOOM_FACTOR
    ax.set(xlim=(x0-r,x0+r),ylim=(y0-r,y0+r),xlabel="X / Easting",ylabel="Y / Northing",
           title=f"Neighborhood and Kriging Weights — {d['model']}")
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(fontsize=7,loc="upper right")
    err="" if not np.isfinite(observed) else f"\nObserved = {observed:.3f}\nError = {d['pred']-observed:+.3f}"
    info.text(.02,.98,f"Model = {d['model']}\nPrediction = {d['pred']:.3f}{err}\nKriging variance = {d['var']:.3f}\n"
                      f"Candidates ≤400 m = {d['n_inside']}\nN selected = {len(idx)}\nN negative = {np.sum(w<0)}\n"
                      f"Sum λ = {w.sum():.6f}\ncond(K) = {d['cond']:.3e}",va="top",fontsize=9,
              bbox=dict(boxstyle="round",facecolor="white",edgecolor="gray"))
    show=d["table"][["N","ID","Z","h","R","Gamma","Lambda","Lambda_Z"]].copy()
    for c,nf in [("Z",2),("h",1),("R",1),("Gamma",2),("Lambda",4),("Lambda_Z",3)]:
        show[c]=show[c].map(lambda v,f=nf:f"{v:.{f}f}")
    tbl=at.table(cellText=show.values,colLabels=show.columns,cellLoc="center",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.2); tbl.scale(1,1.3)
    at.set_title(f"Selected Neighbours — {d['model']}")
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUTPUT_DIR, f"diagnostic_neighborhood_{d['model']}.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()


def run_neighborhood_diagnostics():
    x0,y0,obs,tr,ztr,ids=diagnostic_target(); out={}
    print("="*70,"\nDIAGNOSTIK SEARCH NEIGHBORHOOD: CIRCLE 400 m\n"+"="*70,sep="")
    for m in MODELS:
        d=solve_diagnostic(x0,y0,tr,ztr,m,ids)
        if d is None: print(f"{m}: neighbour dalam 400 m tidak mencukupi."); continue
        out[m]=d; plot_diagnostic(d,obs)
        print(f"{m}: prediction={d['pred']:.4f}, n={len(d['idx'])}, negative={np.sum(d['w']<0)}")
    return out


neighborhood_diagnostic_output = run_neighborhood_diagnostics() if RUN_NEIGHBOR_DIAGNOSTIC else None

# ============================================================
# 9. MEMBUAT BOUNDARY ORTOGONAL DAN GRID ESTIMASI
# ============================================================
class OrthogonalBoundary:
    """Boundary ortogonal tanpa Shapely, dibentuk dari union kotak ±offset."""
    def __init__(self, x_edges, y_edges, mask):
        self.x_edges = np.asarray(x_edges, float)
        self.y_edges = np.asarray(y_edges, float)
        self.mask = np.asarray(mask, bool)
        self.bounds = (self.x_edges[0], self.y_edges[0],
                       self.x_edges[-1], self.y_edges[-1])
        dx, dy = np.diff(self.x_edges), np.diff(self.y_edges)
        self.area = float(np.sum(self.mask * dy[:, None] * dx[None, :]))
        self.segments = self._segments()

    def _segments(self):
        seg, m, xe, ye = [], self.mask, self.x_edges, self.y_edges
        ny, nx = m.shape
        for i, j in np.argwhere(m):
            if j == 0 or not m[i, j-1]:
                seg.append(((xe[j], ye[i]), (xe[j], ye[i+1])))
            if j == nx-1 or not m[i, j+1]:
                seg.append(((xe[j+1], ye[i]), (xe[j+1], ye[i+1])))
            if i == 0 or not m[i-1, j]:
                seg.append(((xe[j], ye[i]), (xe[j+1], ye[i])))
            if i == ny-1 or not m[i+1, j]:
                seg.append(((xe[j], ye[i+1]), (xe[j+1], ye[i+1])))
        return seg


def build_orthogonal_boundary(xy, offset, cell=BOUNDARY_CELL):
    """
    Boundary ortogonal yang lebih rapi memakai raster reguler 25 m.
    Dasar boundary tetap union kotak ±offset dari tiap titik bor,
    kemudian dilakukan closing dan fill holes agar tidak bolong.
    """
    xy = np.asarray(xy, float)
    minx = np.floor((xy[:, 0].min() - offset) / cell) * cell
    maxx = np.ceil((xy[:, 0].max() + offset) / cell) * cell
    miny = np.floor((xy[:, 1].min() - offset) / cell) * cell
    maxy = np.ceil((xy[:, 1].max() + offset) / cell) * cell

    x_edges = np.arange(minx, maxx + cell, cell, dtype=float)
    y_edges = np.arange(miny, maxy + cell, cell, dtype=float)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    mask = np.zeros((ny, nx), dtype=bool)

    for x, y in xy:
        ix0 = max(0, np.searchsorted(x_edges, x - offset, side="right") - 1)
        ix1 = min(nx, np.searchsorted(x_edges, x + offset, side="left"))
        iy0 = max(0, np.searchsorted(y_edges, y - offset, side="right") - 1)
        iy1 = min(ny, np.searchsorted(y_edges, y + offset, side="left"))
        mask[iy0:iy1, ix0:ix1] = True

    if BOUNDARY_CLOSE_ITERS > 0:
        mask = binary_closing(mask, structure=np.ones((3, 3), bool), iterations=BOUNDARY_CLOSE_ITERS)

    if FILL_INTERNAL_BOUNDARY_HOLES:
        mask = binary_fill_holes(mask)

    return OrthogonalBoundary(x_edges, y_edges, mask)


def points_inside_geometry(g, xy):
    """Uji titik terhadap raster-cell boundary; titik pada garis boundary ikut diterima."""
    xy = np.asarray(xy, float); x, y = xy[:, 0], xy[:, 1]
    ix_a = np.searchsorted(g.x_edges, x, side="right") - 1
    ix_b = np.searchsorted(g.x_edges, x, side="left") - 1
    iy_a = np.searchsorted(g.y_edges, y, side="right") - 1
    iy_b = np.searchsorted(g.y_edges, y, side="left") - 1
    inside = np.zeros(len(xy), dtype=bool)
    ny, nx = g.mask.shape
    for ix in (ix_a, ix_b):
        for iy in (iy_a, iy_b):
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            if np.any(ok): inside[ok] |= g.mask[iy[ok], ix[ok]]
    return inside


estimation_boundary = build_orthogonal_boundary(coords, BOUNDARY_OFFSET, BOUNDARY_CELL)
minx, miny, maxx, maxy = estimation_boundary.bounds
xmin, xmax = np.floor(minx/GRID_RES)*GRID_RES, np.ceil(maxx/GRID_RES)*GRID_RES
ymin, ymax = np.floor(miny/GRID_RES)*GRID_RES, np.ceil(maxy/GRID_RES)*GRID_RES
x_grid = np.arange(xmin, xmax+GRID_RES, GRID_RES)
y_grid = np.arange(ymin, ymax+GRID_RES, GRID_RES)
XX, YY = np.meshgrid(x_grid, y_grid)
grid_points = np.c_[XX.ravel(), YY.ravel()]
inside_mask = (points_inside_geometry(estimation_boundary, grid_points)
               if MASK_OUTSIDE_BOUNDARY else np.ones(len(grid_points), bool))
print("="*70, "\nGRID ESTIMASI\n"+"="*70, sep="")
print(f"Boundary offset = {BOUNDARY_OFFSET:.1f} m | cell = {BOUNDARY_CELL:.1f} m | bentuk = ortogonal rapi | luas = {estimation_boundary.area:.1f} m²")
print(f"Grid total = {len(grid_points)} | diestimasi = {inside_mask.sum()} | lubang internal = diisi\n")

# ============================================================
# 10. ESTIMASI GRID UNTUK 2 MODEL
# ============================================================
grid_estimates={}; grid_variances_raw={}; grid_nneighbors={}; grid_nnegative_weights={}
valid_idx=np.where(inside_mask)[0]
for model in MODELS:
    print(f"Estimasi grid model: {model}")
    zf=np.full(len(grid_points),np.nan); vf=zf.copy(); nf=zf.copy(); ng=zf.copy()
    for count,gi in enumerate(valid_idx,1):
        out=ordinary_kriging_point(*grid_points[gi],coords,z_data,model)
        zf[gi],vf[gi],nf[gi],ng[gi]=out[:4]
        if count%500==0: print(f"  selesai {count}/{len(valid_idx)} grid")
    grid_estimates[model]=zf.reshape(XX.shape); grid_variances_raw[model]=vf.reshape(XX.shape)
    grid_nneighbors[model]=nf.reshape(XX.shape); grid_nnegative_weights[model]=ng.reshape(XX.shape)
print("Estimasi grid selesai.\n")

# ============================================================
# 11. EXPORT HASIL KRIGING LIM KE EXCEL
#     Langsung setelah estimasi grid selesai
# ============================================================

# Gabungkan hasil 2 model ke satu tabel utama
kriging_export_df = pd.DataFrame({
    "X": XX.ravel(),
    "Y": YY.ravel(),
    "Inside_Boundary": inside_mask
})

for m in MODELS:
    kriging_export_df[f"Estimated_LIM_{m}"] = grid_estimates[m].ravel()

# Hanya grid di dalam boundary yang dipakai
kriging_export_inside_df = kriging_export_df.loc[
    kriging_export_df["Inside_Boundary"]
].copy()

# Hapus baris jika seluruh model gagal menghasilkan estimasi
est_cols = [f"Estimated_LIM_{m}" for m in MODELS]
kriging_export_inside_df = kriging_export_inside_df.dropna(
    subset=est_cols,
    how="all"
).reset_index(drop=True)

excel_path = os.path.join(
    OUTPUT_DIR,
    "hasil_kriging_LIM_XY_2model.xlsx"
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    # Sheet utama: X-Y + hasil kedua model
    kriging_export_inside_df.to_excel(
        writer,
        sheet_name="Kriging_LIM",
        index=False
    )

    # Sheet terpisah masing-masing model
    for m in MODELS:
        tmp = kriging_export_inside_df[
            ["X", "Y", f"Estimated_LIM_{m}"]
        ].copy()

        tmp.columns = [
            "X",
            "Y",
            "Estimated_LIM_m"
        ]

        tmp.to_excel(
            writer,
            sheet_name=f"LIM_{m}"[:31],
            index=False
        )

print("\n" + "="*70)
print("EXCEL HASIL ESTIMASI KRIGING LIM BERHASIL DIBUAT")
print("="*70)
print(f"File : {os.path.abspath(excel_path)}")
print(f"Jumlah grid di dalam boundary : {len(kriging_export_inside_df)}")
print("\nSheet yang tersedia:")
print("  1. Kriging_LIM")
print("  2. LIM_Elips")
print("  3. LIM_Cassini")
print("\nKolom sheet utama:")
print("  X")
print("  Y")
print("  Inside_Boundary")
print("  Estimated_LIM_Elips")
print("  Estimated_LIM_Cassini")
print("\nPROGRAM SELESAI NORMAL — tidak menjalankan statistik, peta, atau LOOCV.")
