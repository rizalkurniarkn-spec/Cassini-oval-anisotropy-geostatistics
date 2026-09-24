# ============================================================

# Models: 1) Ellipse | 2) Affine-scaled Cassini Oval
#
# Public-repository version. The default input is a synthetic example dataset.
# Replace DATA_FILE with an authorized local dataset to reproduce private-data runs.
#






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

# ============================================================
DATA_FILE, SHEET_NAME = str(REPO_ROOT / "example" / "synthetic_thickness.xlsx"), 0
X_COL, Y_COL, Z_COL = None, None, "Saprolite Thickness"
GRID_RES = 12.5
DRILL_SPACING = 25.0
BOUNDARY_OFFSET = DRILL_SPACING
FILL_INTERNAL_BOUNDARY_HOLES = True
MASK_OUTSIDE_BOUNDARY = True
BOUNDARY_CELL = DRILL_SPACING        
BOUNDARY_CLOSE_ITERS = 1             # close small boundary gaps and indentations


SEARCH_RADIUS = 400.0            # meter
NMIN, NMAX = 6, 12
ALLOW_FEWER_WITHIN_RADIUS = True 
MIN_FALLBACK_NEIGHBORS = 1       

NUGGET, PARTIAL_SILL = 22.86, 9.67
SILL = NUGGET + PARTIAL_SILL
EPS = 1e-10
SAVE_FIGS, EXPORT_EXCEL = True, True

EXPORT_KRIGING_EARLY = True
STOP_AFTER_KRIGING_EXPORT = True   


OUTPUT_DIR = str(REPO_ROOT / "outputs" / "ordinary_kriging_saprolite")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_NEIGHBOR_DIAGNOSTIC = True
DIAGNOSTIC_TARGET_MODE = "sample_index"  # sample_index | coordinate
DIAGNOSTIC_SAMPLE_INDEX = 0
DIAGNOSTIC_X = DIAGNOSTIC_Y = None
DIAGNOSTIC_SHOW_LABELS = True
DIAGNOSTIC_ZOOM_FACTOR = 1.12

# ============================================================

# ============================================================
ELLIPSE_MAJOR, ELLIPSE_MINOR, ELLIPSE_PHI_AZ = 176.11040, 98.47138, 67.69179
CASSINI_A, CASSINI_B = 118.67893, 140.04114
_CASSINI_INTERNAL_SCALE_X, _CASSINI_INTERNAL_SCALE_Y, CASSINI_PHI_AZ = 0.89533, 1.11691, 65.60143
MODELS = ["Ellipse", "Cassini"]

parameter_model_df = pd.DataFrame([
    {"Model":"Ellipse", "Range_m":np.nan, "Major_m":ELLIPSE_MAJOR, "Minor_m":ELLIPSE_MINOR, "Azimuth_deg":ELLIPSE_PHI_AZ, "Cassini_a_m":np.nan, "Cassini_b_m":np.nan},
    {"Model":"Cassini", "Range_m":np.nan, "Major_m":np.nan, "Minor_m":np.nan, "Azimuth_deg":CASSINI_PHI_AZ, "Cassini_a_m":CASSINI_A, "Cassini_b_m":CASSINI_B}
])

print("=" * 70, "\nESTIMATION PARAMETERS\n" + "=" * 70, sep="")
print(f"GRID_RES              = {GRID_RES} m")
print(f"BOUNDARY_OFFSET       = {BOUNDARY_OFFSET} m")
print(f"SEARCH_RADIUS         = {SEARCH_RADIUS} m (circular, both models)")
print(f"NMIN / NMAX           = {NMIN} / {NMAX}")
print(f"NUGGET / PARTIAL SILL = {NUGGET} / {PARTIAL_SILL}")
print("Anisotropy is applied in the variogram, not in the search neighborhood.")
print(f"Output folder          = {os.path.abspath(OUTPUT_DIR)}\n")

# ============================================================

# ============================================================
def find_excel_file(name):
    if os.path.exists(name): return name
    c = glob.glob("*Thickness*.xlsx") + glob.glob("*Thickness*.xls")
    if c: return c[0]
    raise FileNotFoundError("Input workbook was not found.")


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
    zc = Z_COL if Z_COL in df.columns else detect_column(df, ["saprolite thickness", "thickness", "z"])
    if xc is None or yc is None or zc is None:
        raise ValueError(f"X/Y/Z columns could not be detected. Available columns: {df.columns.tolist()}")

    d = df[[xc, yc, zc]].copy(); d.columns = ["X", "Y", "Z"]
    d[["X", "Y"]] = d[["X", "Y"]].apply(pd.to_numeric, errors="coerce")
    d["Z"] = pd.to_numeric(d["Z"], errors="coerce").fillna(0.0)  
    d = d.dropna(subset=["X", "Y"]).reset_index(drop=True)

    print("=" * 70, "\nDATA LOADED SUCCESSFULLY\n" + "=" * 70, sep="")
    print(f"File = {path} | X = {xc} | Y = {yc} | Z = {zc}")
    print(f"Number of data = {len(d)} | Z min/max/mean = {d.Z.min():.3f} / {d.Z.max():.3f} / {d.Z.mean():.3f}\n")
    return d


data = load_thickness_data()
coords = data[["X", "Y"]].values
x_data, y_data, z_data = data.X.values, data.Y.values, data.Z.values

# ============================================================

# ============================================================
def azimuth_to_unit_vector(az):
    a = np.deg2rad(np.asarray(az, float)); return np.sin(a), np.cos(a)


def pair_azimuth_from_dxdy(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def phi_az_to_theta_math(phi):
    return np.deg2rad((90.0 - phi) % 360.0)

# ============================================================

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
    X = (ux*ct + uy*st) / _CASSINI_INTERNAL_SCALE_X
    Y = (-ux*st + uy*ct) / _CASSINI_INTERNAL_SCALE_Y
    S = X**2 + Y**2
    qa = S**2
    qb = CASSINI_A**2 * (2*S - 4*X**2)
    qc = CASSINI_A**4 - CASSINI_B**4
    disc = np.maximum(qb**2 - 4*qa*qc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q1 = (-qb + np.sqrt(disc)) / (2*qa)
        q2 = (-qb - np.sqrt(disc)) / (2*qa)
    q = np.nanmax(np.where(np.stack([q1, q2]) > 0, np.stack([q1, q2]), np.nan), axis=0)
    return np.sqrt(q)


def get_directional_range(az, model):
    return {"Ellipse": ellipse_range_from_azimuth,
            "Cassini": cassini_range_from_azimuth}[model](az)

# ============================================================

# ============================================================
def spherical_variogram_from_dxdy(dx, dy, model):
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    h = np.hypot(dx, dy)
    a = get_directional_range(pair_azimuth_from_dxdy(dx, dy), model)
    if np.any(~np.isfinite(a) | (a <= EPS)):
        raise ValueError(f"Invalid directional range: {model}")
    t = h / a
    return np.where(h <= EPS, 0.0,
                    np.where(t <= 1.0, NUGGET + PARTIAL_SILL*(1.5*t - 0.5*t**3), SILL))

# ============================================================

# ============================================================
def get_neighbors(x0, y0, coords_train):
    """Both models use a Euclidean search circle <= 400 m with at most NMAX samples."""
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
    print("="*70,"\nSEARCH-NEIGHBORHOOD DIAGNOSTIC: CIRCLE 400 m\n"+"="*70,sep="")
    for m in MODELS:
        d=solve_diagnostic(x0,y0,tr,ztr,m,ids)
        if d is None: print(f"{m}: neighbors within 400 m are insufficient."); continue
        out[m]=d; plot_diagnostic(d,obs)
        print(f"{m}: prediction={d['pred']:.4f}, n={len(d['idx'])}, negative={np.sum(d['w']<0)}")
    return out


neighborhood_diagnostic_output = run_neighborhood_diagnostics() if RUN_NEIGHBOR_DIAGNOSTIC else None

# ============================================================

# ============================================================
class OrthogonalBoundary:
    """Build an orthogonal boundary from the union of offset boxes without Shapely."""
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
    A regular 25 m raster is used to construct a cleaner orthogonal boundary.
    The boundary remains the union of offset boxes around each drillhole,
    followed by closing and hole filling.
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
    """Test points against the raster-cell boundary; points on the boundary are included."""
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
print("="*70, "\nESTIMATION GRID\n"+"="*70, sep="")
print(f"Boundary offset = {BOUNDARY_OFFSET:.1f} m | cell = {BOUNDARY_CELL:.1f} m | shape = orthogonal | area = {estimation_boundary.area:.1f} m²")
print(f"Grid total = {len(grid_points)} | estimated = {inside_mask.sum()} | internal holes = filled\n")

# ============================================================

# ============================================================
grid_estimates={}; grid_variances_raw={}; grid_nneighbors={}; grid_nnegative_weights={}
valid_idx=np.where(inside_mask)[0]
for model in MODELS:
    print(f"Estimating grid with model: {model}")
    zf=np.full(len(grid_points),np.nan); vf=zf.copy(); nf=zf.copy(); ng=zf.copy()
    for count,gi in enumerate(valid_idx,1):
        out=ordinary_kriging_point(*grid_points[gi],coords,z_data,model)
        zf[gi],vf[gi],nf[gi],ng[gi]=out[:4]
        if count%500==0: print(f"  completed {count}/{len(valid_idx)} grid nodes")
    grid_estimates[model]=zf.reshape(XX.shape); grid_variances_raw[model]=vf.reshape(XX.shape)
    grid_nneighbors[model]=nf.reshape(XX.shape); grid_nnegative_weights[model]=ng.reshape(XX.shape)
print("Grid estimation completed.\n")

# ============================================================

# ============================================================


if EXPORT_KRIGING_EARLY:
    kriging_export_df = pd.DataFrame({
        "X": XX.ravel(),
        "Y": YY.ravel(),
        "Inside_Boundary": inside_mask
    })

    for m in MODELS:
        kriging_export_df[f"Estimated_SAP_{m}"] = grid_estimates[m].ravel()

    
    kriging_export_inside_df = kriging_export_df.loc[
        kriging_export_df["Inside_Boundary"]
    ].copy()

    
    est_cols = [f"Estimated_SAP_{m}" for m in MODELS]
    kriging_export_inside_df = kriging_export_inside_df.dropna(
        subset=est_cols, how="all"
    ).reset_index(drop=True)

    kriging_excel_path = os.path.join(
        OUTPUT_DIR,
        "kriging_saprolite_XY_2models.xlsx"
    )

    with pd.ExcelWriter(kriging_excel_path, engine="openpyxl") as writer:
        
        kriging_export_inside_df.to_excel(
            writer, sheet_name="Kriging_SAP", index=False
        )

        
        for m in MODELS:
            tmp = kriging_export_inside_df[[
                "X", "Y", f"Estimated_SAP_{m}"
            ]].copy()
            tmp.columns = ["X", "Y", "Estimated_SAP_m"]
            tmp.to_excel(
                writer, sheet_name=f"SAP_{m}"[:31], index=False
            )

    print("=" * 70)
    print("SAPROLITE KRIGING ESTIMATE WORKBOOK CREATED")
    print("=" * 70)
    print(f"File : {os.path.abspath(kriging_excel_path)}")
    print(f"Number of grid nodes inside boundary : {len(kriging_export_inside_df)}")
    print("Main columns:")
    print("  X")
    print("  Y")
    for m in MODELS:
        print(f"  Estimated_SAP_{m}")
    print()

    if STOP_AFTER_KRIGING_EXPORT:
        print("STOP_AFTER_KRIGING_EXPORT = True")
        print("Program stopped after the kriging-estimate workbook was created.")
        print("Set STOP_AFTER_KRIGING_EXPORT=False to continue with statistics, maps, and LOOCV.")
        raise SystemExit

# ============================================================
# 11. STATISTIK DESKRIPTIF
# ============================================================
def descriptive_stats(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if not len(v): return {k:np.nan for k in ["N","Min","Max","Mean","Median","Std","CV","P05","P25","P75","P95"]}
    mean,std=v.mean(),v.std(ddof=1)
    return dict(N=len(v),Min=v.min(),Max=v.max(),Mean=mean,Median=np.median(v),Std=std,
                CV=std/mean if abs(mean)>EPS else np.nan,P05=np.percentile(v,5),P25=np.percentile(v,25),
                P75=np.percentile(v,75),P95=np.percentile(v,95))

rows=[]
for name,v in [("Data Aktual",z_data)]+[(m,grid_estimates[m].ravel()) for m in MODELS]:
    r=descriptive_stats(v); r["Model"]=name; rows.append(r)
global_stats_df=pd.DataFrame(rows)[["Model","N","Min","Max","Mean","Median","Std","CV","P05","P25","P75","P95"]]
print("="*70,"\nSTATISTIK GLOBAL\n"+"="*70,sep=""); print(global_stats_df.round(4).to_string(index=False))

# ============================================================

# ============================================================
cmap_smooth=LinearSegmentedColormap.from_list("smooth_blue_red",
    ["#0b3c8c","#2c7fb8","#41b6c4","#a1dab4","#ffffbf","#fdae61","#d7191c"],N=256)
N_LEVELS_MAP=30; POINT_SIZE_SINGLE=24; POINT_SIZE_COMPARE=18
COLOR_SCALE_SINGLE, COLOR_SCALE_COMPARE="local","global"


def plot_estimation_boundary(ax, g, label=None, lw=2):
    lc = LineCollection(g.segments, colors="black", linewidths=lw,
                        zorder=10, label=label)
    ax.add_collection(lc)


def map_panel(ax,model,vmin,vmax,point_size=20,boundary_label=None):
    cf=ax.contourf(XX,YY,np.ma.masked_invalid(grid_estimates[model]),levels=N_LEVELS_MAP,
                   cmap=cmap_smooth,vmin=vmin,vmax=vmax)
    ax.scatter(x_data,y_data,c=z_data,cmap=cmap_smooth,s=point_size,edgecolor="k",lw=.35,vmin=vmin,vmax=vmax)
    plot_estimation_boundary(ax,estimation_boundary,boundary_label,1.7)
    ax.set(xlabel="X",ylabel="Y"); ax.set_aspect("equal"); ax.grid(alpha=.18)
    return cf

# ============================================================

# ============================================================
allvals=np.concatenate([grid_estimates[m].ravel() for m in MODELS]); allvals=allvals[np.isfinite(allvals)]
for m in MODELS:
    Z=grid_estimates[m]; vmin,vmax=(np.nanmin(Z),np.nanmax(Z)) if COLOR_SCALE_SINGLE=="local" else (allvals.min(),allvals.max())
    fig,ax=plt.subplots(figsize=(10,8),constrained_layout=True)
    cf=map_panel(ax,m,vmin,vmax,POINT_SIZE_SINGLE,f"Estimation boundary (±{BOUNDARY_OFFSET:.0f} m)")
    fig.colorbar(cf,ax=ax,label="Estimated Saprolite Thickness"); ax.set_title(f"{m} - Estimasi Saprolite Thickness",fontweight="bold"); ax.legend(fontsize=8)
    if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,f"map_estimation_{m}.png"),dpi=300,bbox_inches="tight")
    plt.show()

# ============================================================

# ============================================================
vminG,vmaxG=allvals.min(),allvals.max(); fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
for ax,m in zip(axes,MODELS):
    Z=grid_estimates[m]; vmin,vmax=(np.nanmin(Z),np.nanmax(Z)) if COLOR_SCALE_COMPARE=="local" else (vminG,vmaxG)
    cf=map_panel(ax,m,vmin,vmax,POINT_SIZE_COMPARE); ax.set_title(f"Model {m}",fontweight="bold")
fig.colorbar(cf,ax=axes.ravel().tolist(),label="Estimated Saprolite Thickness")
if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,"map_comparison_2_models.png"),dpi=300,bbox_inches="tight")
plt.show()

# ============================================================

# ============================================================
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for ax,m in zip(axes,MODELS):
    v=grid_estimates[m].ravel(); v=v[np.isfinite(v)]
    ax.hist(v,bins=20,edgecolor="k",alpha=.85); ax.axvline(v.mean(),ls="--",lw=2,label=f"Mean = {v.mean():.2f}")
    ax.axvline(np.median(v),lw=2,label=f"Median = {np.median(v):.2f}"); ax.set(title=f"Histogram of Estimates\nModel {m}",xlabel="Saprolite Thickness",ylabel="Frequency"); ax.grid(alpha=.25); ax.legend(fontsize=8)
if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,"histogram_estimates_2_models.png"),dpi=300,bbox_inches="tight")
plt.show()

# ============================================================
# 16. CROSS VALIDATION LEAVE-ONE-OUT
# ============================================================
def r2_manual(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); den=np.sum((y-y.mean())**2)
    return np.nan if den<=EPS else 1-np.sum((y-p)**2)/den


def leave_one_out_cv():
    rows=[]; n=len(data)
    for i in range(n):
        if (i+1)%25==0: print(f"Cross-validation point {i+1}/{n}")
        mask=np.ones(n,bool); mask[i]=False
        row=dict(Index=i,X=x_data[i],Y=y_data[i],Z_actual=z_data[i])
        for m in MODELS:
            p,v,nn,nneg,wmin,wmax=ordinary_kriging_point(x_data[i],y_data[i],coords[mask],z_data[mask],m)
            row.update({f"Z_pred_{m}":p,f"Error_{m}":p-z_data[i],f"KrigingVariance_{m}":v,
                        f"N_used_{m}":nn,f"N_negative_weights_{m}":nneg,f"Min_weight_{m}":wmin,f"Max_weight_{m}":wmax})
        rows.append(row)
    return pd.DataFrame(rows)

print("="*70,"\nMULAI LEAVE-ONE-OUT CROSS VALIDATION\n"+"="*70,sep="")
cv_df=leave_one_out_cv(); print("Cross-validation completed.\n")

# ============================================================
# 17. CROSS-VALIDATION SUMMARY
# ============================================================
rows=[]
for m in MODELS:
    valid=np.isfinite(cv_df[f"Z_pred_{m}"]); a=cv_df.loc[valid,"Z_actual"].values; p=cv_df.loc[valid,f"Z_pred_{m}"].values; e=p-a
    rows.append(dict(Model=m,N_valid=len(a),ME=e.mean(),MAE=np.abs(e).mean(),RMSE=np.sqrt(np.mean(e**2)),
                     R2=r2_manual(a,p),Std_Error=e.std(ddof=1),Min_Error=e.min(),Max_Error=e.max(),
                     Mean_Negative_Weights=cv_df.loc[valid,f"N_negative_weights_{m}"].mean()))
cv_summary_df=pd.DataFrame(rows)
print("="*70,"\nCROSS-VALIDATION SUMMARY\n"+"="*70,sep=""); print(cv_summary_df.round(4).to_string(index=False))

# ============================================================
# 18. SCATTER ACTUAL VS PREDICTED + REGRESSION LINE
# ============================================================
def regression_summary(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    if len(x)<2:return dict(R=np.nan,R2=np.nan,a=np.nan,b=np.nan,RMSE=np.nan)
    b,a=np.polyfit(x,y,1); return dict(R=np.corrcoef(x,y)[0,1],R2=r2_manual(x,y),a=a,b=b,RMSE=np.sqrt(np.mean((y-x)**2)))

pred_all=[cv_df[f"Z_pred_{m}"].dropna().values for m in MODELS]
lo=min(np.nanmin(z_data),*[v.min() for v in pred_all if len(v)]); hi=max(np.nanmax(z_data),*[v.max() for v in pred_all if len(v)])
r=max(hi-lo,1); lo=max(0,lo-.05*r); hi+=.05*r
fig,axes=plt.subplots(1,2,figsize=(14,5.8),constrained_layout=True)
for ax,m in zip(axes,MODELS):
    ok=np.isfinite(cv_df[f"Z_pred_{m}"]); a=cv_df.loc[ok,"Z_actual"].values; p=cv_df.loc[ok,f"Z_pred_{m}"].values; s=regression_summary(a,p)
    ax.scatter(a,p,s=22,alpha=.75); ax.plot([lo,hi],[lo,hi],"--",lw=1.5,label="1:1 line")
    xx=np.linspace(lo,hi,200); ax.plot(xx,s["a"]+s["b"]*xx,lw=1.5,label="Regression line")
    ax.text(.05,.95,f"R = {s['R']:.3f}\nR² = {s['R2']:.3f}\na = {s['a']:.3f}\nb = {s['b']:.3f}\nRMSE = {s['RMSE']:.3f}",
            transform=ax.transAxes,va="top",fontweight="bold",bbox=dict(boxstyle="round",facecolor="white",alpha=.9))
    ax.set(xlim=(lo,hi),ylim=(lo,hi),title=f"Cross Validation Scatter Plot - {m}",xlabel=f"Observed {Z_COL}",ylabel=f"Predicted {Z_COL}"); ax.grid(alpha=.25); ax.legend(fontsize=8)
if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,"scatter_cv_2_model_revised.png"),dpi=300,bbox_inches="tight")
plt.show()

# ============================================================
# 19. HISTOGRAM ERROR CROSS VALIDATION
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for ax,m in zip(axes,MODELS):
    e=cv_df[f"Error_{m}"].dropna().values; ax.hist(e,bins=20,edgecolor="k",alpha=.85); ax.axvline(0,lw=2,label="Error = 0"); ax.axvline(e.mean(),ls="--",lw=2,label=f"ME = {e.mean():.2f}")
    ax.set(title=f"Cross-validation Error Histogram\nModel {m}",xlabel="Prediction error",ylabel="Frequency"); ax.grid(alpha=.25); ax.legend(fontsize=8)
if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,"histogram_error_cv_2_model.png"),dpi=300,bbox_inches="tight")
plt.show()

# ============================================================
# 20. BAR CHART METRIK CROSS VALIDATION
# ============================================================
fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
for ax,metric in zip(axes,["RMSE","MAE","ME"]):
    vals=cv_summary_df[metric]; ax.bar(cv_summary_df.Model,vals,edgecolor="k",alpha=.85); ax.set(title=f"Cross-validation comparison: {metric}",ylabel=metric); ax.grid(axis="y",alpha=.25)
    for i,v in enumerate(vals): ax.text(i,v,f"{v:.2f}",ha="center",va="bottom")
if SAVE_FIGS: fig.savefig(os.path.join(OUTPUT_DIR,"bar_metric_cv_2_model.png"),dpi=300,bbox_inches="tight")
plt.show()

# ============================================================
# 21. DATAFRAME OUTPUT
# ============================================================
grid_result_df=pd.DataFrame({"X":XX.ravel(),"Y":YY.ravel(),"Inside_Boundary":inside_mask,"Estimated":inside_mask})
for m in MODELS:
    grid_result_df[f"Z_{m}"]=grid_estimates[m].ravel()
    grid_result_df[f"KrigingVarianceRaw_{m}"]=grid_variances_raw[m].ravel()
    grid_result_df[f"N_neighbors_{m}"]=grid_nneighbors[m].ravel()
    grid_result_df[f"N_negative_raw_weights_{m}"]=grid_nnegative_weights[m].ravel()


def boundary_vertices_dataframe(g):
    rows=[]
    for pid, seg in enumerate(g.segments, 1):
        for vid, (x, y) in enumerate(seg, 1):
            rows.append(dict(Part=pid, Vertex=vid, X=x, Y=y))
    return pd.DataFrame(rows)

boundary_vertices_df=boundary_vertices_dataframe(estimation_boundary)
print("="*70,"\nPYTHON OUTPUT VARIABLES\n"+"="*70,sep="")
print("1. global_stats_df\n2. cv_summary_df\n3. cv_df\n4. grid_result_df\n5. grid_estimates\n6. grid_variances_raw\n7. grid_nnegative_weights\n8. neighborhood_diagnostic_output\n9. estimation_boundary\n10. boundary_vertices_df\n11. parameter_model_df")

# ============================================================

# ============================================================
if EXPORT_EXCEL:
    excel_path = os.path.join(OUTPUT_DIR, "saprolite_estimates_2models_boundary25_search400_circle.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        parameter_model_df.to_excel(writer,sheet_name="model_parameters",index=False)
        global_stats_df.to_excel(writer,sheet_name="global_stats",index=False)
        cv_summary_df.to_excel(writer,sheet_name="cv_summary",index=False)
        cv_df.to_excel(writer,sheet_name="cv_detail",index=False)
        grid_result_df.to_excel(writer,sheet_name="grid_estimation",index=False)
        boundary_vertices_df.to_excel(writer,sheet_name="boundary_vertices",index=False)


print("\n" + "="*70)
print("OUTPUT COMPLETE")
print("="*70)
print(f"All output files were saved in:\n{os.path.abspath(OUTPUT_DIR)}")
