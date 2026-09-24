"""NLS fitting of Gaussian-transformed limonite directional ranges.

The script compares the Ellipse and Cassini Oval anisotropy geometries used in
this study. The published Cassini parameters are ``a``, ``b``, and the focus-
axis azimuth. An internal coordinate-normalization term is optimized only to
stabilize the computational representation and is not reported as a geometric
parameter.

Run from a terminal:
    python src/nls_fitting_limonite_gaussian.py --no-show

Dependencies: numpy, pandas, scipy, matplotlib.
Geological azimuth convention: 0 degrees = North, 90 degrees = East.
"""

from itertools import product
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def resolve_repo_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "src").is_dir() and (candidate / "example").is_dir():
            return candidate
    return cwd


REPO_ROOT = resolve_repo_root()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nls_limonite_gaussian"
AZIMUTH_DEG = np.arange(0.0, 360.0, 22.5)
OBSERVED_RANGE_M = np.tile([166., 200., 241., 282., 276., 227., 170., 154.], 2)


def geological_to_math_angle(azimuth_deg):
    return np.deg2rad((90.0 - np.asarray(azimuth_deg, dtype=float)) % 360.0)


def polar_to_xy(azimuth_deg, radius):
    theta = geological_to_math_angle(azimuth_deg)
    return radius * np.cos(theta), radius * np.sin(theta)


def fit_statistics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - observed.mean())**2))
    return {
        "SSE_m2": sse,
        "RMSE_m": float(np.sqrt(np.mean(residual**2))),
        "MAE_m": float(np.mean(np.abs(residual))),
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def best_multistart_fit(residual_function, starting_points, bounds, max_nfev=30000):
    best = None
    for initial in starting_points:
        fit = least_squares(
            residual_function,
            initial,
            bounds=bounds,
            loss="linear",
            max_nfev=max_nfev,
        )
        if fit.success and np.isfinite(fit.cost) and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        raise RuntimeError("No converged NLS solution was found.")
    return best


def fit_ellipse(azimuth_deg, observed_range_m):
    theta_obs = geological_to_math_angle(azimuth_deg)

    def radius(theta, major, minor, phi):
        return 1.0 / np.sqrt(
            np.cos(theta - phi) ** 2 / major**2
            + np.sin(theta - phi) ** 2 / minor**2
        )

    def residual(parameters):
        major, minor = np.exp(parameters[:2])
        return radius(theta_obs, major, minor, parameters[2]) - observed_range_m

    starts = (
        [np.log(major), np.log(minor), np.deg2rad(phi)]
        for major, minor, phi in product(
            np.linspace(220.0, 340.0, 7),
            np.linspace(120.0, 220.0, 6),
            np.arange(0.0, 180.0, 10.0),
        )
    )
    fit = best_multistart_fit(
        residual,
        starts,
        ([-10.0, -10.0, -np.pi], [10.0, 10.0, np.pi]),
        max_nfev=20000,
    )
    major, minor = np.exp(fit.x[:2])
    phi = fit.x[2] % np.pi
    if minor > major:
        major, minor, phi = minor, major, (phi + np.pi / 2.0) % np.pi

    def predict(azimuth):
        return radius(geological_to_math_angle(azimuth), major, minor, phi)

    params = {
        "Major_Range_m": float(major),
        "Minor_Range_m": float(minor),
        "Major_Axis_Azimuth_deg": float((90.0 - np.degrees(phi)) % 180.0),
    }
    return predict, params


def cassini_radius(theta, phi, a, b_over_a, internal_scale_log):
    """Return the computational Cassini Oval radius for a given direction."""
    b = b_over_a * a
    scale_x = np.exp(internal_scale_log)
    scale_y = np.exp(-internal_scale_log)
    relative = theta - phi
    qx = np.cos(relative) / scale_x
    qy = np.sin(relative) / scale_y
    qsum = qx**2 + qy**2
    linear_term = 2.0 * a**2 * qsum - 4.0 * a**2 * qx**2
    constant_term = a**4 - b**4
    discriminant = np.maximum(linear_term**2 - 4.0 * qsum**2 * constant_term, 0.0)
    radius_squared = (-linear_term + np.sqrt(discriminant)) / (2.0 * qsum**2)
    return np.sqrt(np.maximum(radius_squared, 0.0))


def fit_cassini(azimuth_deg, observed_range_m):
    theta_obs = geological_to_math_angle(azimuth_deg)

    def decode(parameters):
        phi = parameters[0]
        a = np.exp(parameters[1])
        b_over_a = 1.0 + np.exp(parameters[2])
        internal_scale_log = parameters[3]
        return phi, a, b_over_a, internal_scale_log

    def residual(parameters):
        return cassini_radius(theta_obs, *decode(parameters)) - observed_range_m

    starts = (
        [np.deg2rad(phi), np.log(a), np.log(ratio - 1.0), scale_log]
        for phi, a, ratio, scale_log in product(
            np.arange(0.0, 180.0, 10.0),
            [80.0, 120.0, 160.0, 200.0, 240.0],
            [1.05, 1.20, 1.50, 2.00, 3.00],
            [-0.50, -0.25, 0.0, 0.25, 0.50],
        )
    )
    fit = best_multistart_fit(
        residual,
        starts,
        ([-np.pi, np.log(1e-4), np.log(1e-6), -1.5],
         [ np.pi, np.log(3000.0), np.log(20.0), 1.5]),
    )
    phi, a, b_over_a, internal_scale_log = decode(fit.x)
    phi = phi % np.pi
    b = b_over_a * a

    def predict(azimuth):
        return cassini_radius(
            geological_to_math_angle(azimuth),
            phi,
            a,
            b_over_a,
            internal_scale_log,
        )

    params = {
        "Cassini_a_m": float(a),
        "Cassini_b_m": float(b),
        "Focus_Axis_Azimuth_deg": float((90.0 - np.degrees(phi)) % 180.0),
    }
    return predict, params


def evaluate_model(name, predict, params):
    predicted = np.asarray(predict(AZIMUTH_DEG), dtype=float)
    fine_azimuth = np.linspace(0.0, 360.0, 1441)
    curve = np.asarray(predict(fine_azimuth), dtype=float)
    return {
        "name": name,
        "predict": predict,
        "params": params,
        "predicted": predicted,
        "residual": predicted - OBSERVED_RANGE_M,
        "curve": curve,
        "fine_azimuth": fine_azimuth,
        "stats": {
            **fit_statistics(OBSERVED_RANGE_M, predicted),
            "Rmax_m": float(curve.max()),
            "Rmin_m": float(curve.min()),
            "Rmax_Rmin": float(curve.max() / curve.min()),
        },
    }


def save_outputs(models, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    comparison = pd.DataFrame({"Azimuth_deg": AZIMUTH_DEG, "Observed_Range_m": OBSERVED_RANGE_M})

    for model in models:
        key = model["name"].lower().replace(" ", "_")
        model_dir = output_dir / key
        model_dir.mkdir(exist_ok=True)
        summary_rows.append({"Model": model["name"], **model["params"], **model["stats"]})
        comparison[f"Predicted_{model['name']}_m"] = model["predicted"]
        comparison[f"Residual_{model['name']}_m"] = model["residual"]
        pd.DataFrame([{"Model": model["name"], **model["params"], **model["stats"]}]).to_csv(
            model_dir / "fit_parameters.csv", index=False
        )
        pd.DataFrame({
            "Azimuth_deg": AZIMUTH_DEG,
            "Observed_Range_m": OBSERVED_RANGE_M,
            "Predicted_Range_m": model["predicted"],
            "Residual_m": model["residual"],
        }).to_csv(model_dir / "observed_predicted_residual.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "nls_summary.csv", index=False)
    comparison.to_csv(output_dir / "nls_observed_vs_predicted.csv", index=False)

    obs_x, obs_y = polar_to_xy(AZIMUTH_DEG, OBSERVED_RANGE_M)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    limit = 1.15 * max(OBSERVED_RANGE_M.max(), *(m["curve"].max() for m in models))
    for ax, model in zip(axes, models):
        x, y = polar_to_xy(model["fine_azimuth"], model["curve"])
        ax.plot(np.r_[obs_x, obs_x[0]], np.r_[obs_y, obs_y[0]], "--", lw=1.2, label="Observed range")
        ax.scatter(obs_x, obs_y, s=28, zorder=3)
        ax.plot(x, y, lw=2.0, label="NLS fit")
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="Easting (m)", ylabel="Northing (m)")
        ax.set_title(model["name"])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Gaussian limonite NLS fitting: Ellipse vs Cassini Oval", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "nls_geometry_comparison.png", dpi=250, bbox_inches="tight")
    return summary


def main(output_dir=DEFAULT_OUTPUT_DIR, show=True):
    print("Fitting 1/2: Ellipse ...", flush=True)
    ellipse_predict, ellipse_params = fit_ellipse(AZIMUTH_DEG, OBSERVED_RANGE_M)
    print("Fitting 2/2: Cassini Oval ...", flush=True)
    cassini_predict, cassini_params = fit_cassini(AZIMUTH_DEG, OBSERVED_RANGE_M)
    models = [
        evaluate_model("Ellipse", ellipse_predict, ellipse_params),
        evaluate_model("Cassini_Oval", cassini_predict, cassini_params),
    ]
    summary = save_outputs(models, output_dir)
    print("\nNLS FITTING SUMMARY")
    print(summary[["Model", "RMSE_m", "MAE_m", "R2", "Rmax_Rmin"]].round(5).to_string(index=False))
    print(f"\nOutputs saved to: {Path(output_dir).resolve()}")
    if show:
        plt.show()
    else:
        plt.close("all")
    return summary, {model["name"]: model["predict"] for model in models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLS fitting of Gaussian limonite directional ranges.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(args.out_dir, show=not args.no_show)
