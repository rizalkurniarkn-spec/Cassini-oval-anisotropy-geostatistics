# -*- coding: utf-8 -*-
"""NLS fitting of saprolite directional ranges: Ellipse vs Cassini Oval.

This public-repository script compares two anisotropy geometries using the
same directional-range observations used in the study:
    1. Elliptical anisotropy
    2. Affine-scaled Cassini Oval anisotropy

The Cassini fit retains the original regularized geometric-distance approach,
including translation and affine-scale penalties. No confidential drillhole
coordinates or assay data are included.

Geological azimuth convention:
    0° = North, 90° = East, clockwise.

Run from a terminal:
    python src/nls_fitting_saprolite.py --no-show

Dependencies:
    numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nls_saprolite"

AZIMUTH_DEG = np.arange(0.0, 360.0, 22.5)
OBSERVED_RANGE_M = np.tile([120.0, 126.0, 150.0, 173.0, 157.0, 135.0, 80.0, 89.0], 2)

QMIN = 1.001
QMAX_GRID = [1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34]
SCALE_PENALTIES = [0.05, 0.10, 0.30, 0.60, 1.00]
SHIFT_PENALTIES = [0.05, 0.15, 0.30]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def geological_to_math_angle(azimuth_deg):
    return np.deg2rad((90.0 - np.asarray(azimuth_deg, dtype=float)) % 360.0)


def polar_to_xy(azimuth_deg, radius):
    theta = geological_to_math_angle(azimuth_deg)
    return radius * np.cos(theta), radius * np.sin(theta)


def rotate(x, y, phi):
    return (
        np.cos(phi) * x + np.sin(phi) * y,
        -np.sin(phi) * x + np.cos(phi) * y,
    )


def fit_statistics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    sse = float(np.sum(residual**2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "SSE_m2": sse,
        "RMSE_m": float(np.sqrt(np.mean(residual**2))),
        "MAE_m": float(np.mean(np.abs(residual))),
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
    }


def best_multistart_fit(residual_function, starting_points, bounds, max_nfev=30000):
    """Return the converged least-squares solution with the smallest cost."""
    best = None
    for initial in starting_points:
        fit = least_squares(
            residual_function,
            initial,
            bounds=bounds,
            loss="linear",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )
        if fit.success and np.isfinite(fit.cost) and (best is None or fit.cost < best.cost):
            best = fit

    if best is None:
        raise RuntimeError("No converged NLS solution was found.")
    return best


# -----------------------------------------------------------------------------
# Ellipse model
# -----------------------------------------------------------------------------
def ellipse_radius(theta_rad, major, minor, phi):
    return 1.0 / np.sqrt(
        np.cos(theta_rad - phi) ** 2 / major**2
        + np.sin(theta_rad - phi) ** 2 / minor**2
    )


def fit_ellipse(azimuth_deg, observed_range_m):
    theta_obs = geological_to_math_angle(azimuth_deg)

    def residual(parameters):
        major, minor = np.exp(parameters[:2])
        phi = parameters[2]
        return ellipse_radius(theta_obs, major, minor, phi) - observed_range_m

    starts = (
        [np.log(major0), np.log(minor0), np.deg2rad(phi0)]
        for major0, minor0, phi0 in product(
            np.linspace(130.0, 220.0, 7),
            np.linspace(60.0, 140.0, 6),
            np.arange(0.0, 180.0, 5.0),
        )
    )

    fit = best_multistart_fit(
        residual,
        starts,
        bounds=([0.0, 0.0, -np.pi], [np.log(1000.0), np.log(1000.0), np.pi]),
    )

    major, minor = np.exp(fit.x[:2])
    phi = float(fit.x[2] % np.pi)
    if minor > major:
        major, minor = minor, major
        phi = float((phi + np.pi / 2.0) % np.pi)

    def predict(azimuth):
        return ellipse_radius(geological_to_math_angle(azimuth), major, minor, phi)

    params = {
        "Major_m": float(major),
        "Minor_m": float(minor),
        "Major_Minor_Ratio": float(major / minor),
        "Major_Axis_Azimuth_deg": float((90.0 - np.degrees(phi)) % 180.0),
    }
    return predict, params


# -----------------------------------------------------------------------------
# Affine-scaled Cassini Oval model
# -----------------------------------------------------------------------------
def cassini_implicit(x, y, a, c):
    """Implicit Cassini Oval function F(x,y)=0."""
    return (x * x + y * y + a * a) ** 2 - 4.0 * a * a * x * x - c**4


def unpack_cassini_parameters(parameters, qmax):
    """Map unconstrained NLS parameters to physically interpretable values."""
    q = QMIN + (qmax - QMIN) / (1.0 + np.exp(-parameters[2]))
    a = np.exp(parameters[1])
    return (
        parameters[0],          # phi
        a,
        a * q,                  # c
        q,                      # c/a
        parameters[3],          # x0
        parameters[4],          # y0
        np.exp(parameters[5]),  # sx
        np.exp(parameters[6]),  # sy
    )


def fit_cassini(azimuth_deg, observed_range_m):
    """Fit the regularized affine-scaled Cassini Oval model."""
    scale = float(np.max(observed_range_m))
    xn, yn = polar_to_xy(azimuth_deg, observed_range_m / scale)

    def geometric_residual(parameters, qmax):
        phi, a, c, _, x0, y0, sx, sy = unpack_cassini_parameters(parameters, qmax)
        x_rot, y_rot = rotate(xn - x0, yn - y0, phi)
        x_scaled = x_rot / sx
        y_scaled = y_rot / sy

        gx = 4.0 * x_scaled * (x_scaled**2 + y_scaled**2 - a**2) / sx
        gy = 4.0 * y_scaled * (x_scaled**2 + y_scaled**2 + a**2) / sy

        return cassini_implicit(x_scaled, y_scaled, a, c) / (
            np.hypot(gx, gy) + 1e-12
        )

    bounds = (
        [-np.pi, np.log(0.05), -20.0, -0.25, -0.25, np.log(0.5), np.log(0.5)],
        [ np.pi, np.log(3.00),  20.0,  0.25,  0.25, np.log(2.0), np.log(2.0)],
    )

    winner = None
    best_geometric_rmse = np.inf

    for qmax, lambda_scale, lambda_shift in product(
        QMAX_GRID, SCALE_PENALTIES, SHIFT_PENALTIES
    ):
        def residual(parameters):
            return np.r_[
                geometric_residual(parameters, qmax),
                np.sqrt(lambda_scale) * parameters[5:7],
                np.sqrt(lambda_shift) * parameters[3:5],
            ]

        fraction = np.clip(
            (min(1.2, qmax - 0.001) - QMIN) / (qmax - QMIN),
            1e-6,
            1.0 - 1e-6,
        )

        starts = [
            [
                (np.deg2rad(45.0 + angle) + np.pi) % (2.0 * np.pi) - np.pi,
                np.log(0.7),
                np.log(fraction / (1.0 - fraction)),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            for angle in np.arange(0.0, 180.0, 22.5)
        ]

        fit = best_multistart_fit(residual, starts, bounds)
        geometric_rmse = float(
            np.sqrt(np.mean(geometric_residual(fit.x, qmax) ** 2)) * scale
        )

        if geometric_rmse < best_geometric_rmse:
            winner = (fit.x, qmax, lambda_scale, lambda_shift)
            best_geometric_rmse = geometric_rmse

    if winner is None:
        raise RuntimeError("Cassini Oval fitting did not converge.")

    parameters, qmax, lambda_scale, lambda_shift = winner
    phi, a, c, c_over_a, x0, y0, sx, sy = unpack_cassini_parameters(parameters, qmax)

    def predict(azimuth):
        def ray_root(theta):
            def function(radius):
                x_rot, y_rot = rotate(
                    radius * np.cos(theta) / scale - x0,
                    radius * np.sin(theta) / scale - y0,
                    phi,
                )
                return cassini_implicit(x_rot / sx, y_rot / sy, a, c)

            upper = 4.0 * scale
            if function(0.0) * function(upper) < 0.0:
                return brentq(function, 0.0, upper, xtol=1e-10)

            grid = np.linspace(0.0, upper, 600)
            values = function(grid)
            roots = [
                brentq(function, left, right)
                for left, right, f_left, f_right in zip(
                    grid[:-1], grid[1:], values[:-1], values[1:]
                )
                if f_left * f_right < 0.0
            ]
            roots.extend(grid[values == 0.0])
            if not roots:
                raise ValueError("Cassini Oval does not intersect one requested ray.")
            return max(roots)

        azimuth = np.asarray(azimuth, dtype=float)
        radii = np.array(
            [ray_root(theta) for theta in geological_to_math_angle(azimuth).ravel()]
        ).reshape(azimuth.shape)
        return radii.item() if radii.ndim == 0 else radii

    params = {
        "Cassini_a_m": float(a * scale),
        "Cassini_c_m": float(c * scale),
        "c_over_a": float(c_over_a),
        "sx": float(sx),
        "sy": float(sy),
        "sx_over_sy": float(sx / sy),
        "x0_m": float(x0 * scale),
        "y0_m": float(y0 * scale),
        "Focus_Axis_Azimuth_deg": float((90.0 - np.degrees(phi)) % 180.0),
        "QMAX": float(qmax),
        "lambda_scale": float(lambda_scale),
        "lambda_shift": float(lambda_shift),
        "Geometric_RMSE_approx_m": float(best_geometric_rmse),
    }
    return predict, params


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def evaluate_model(name, predict, params):
    predicted = np.asarray(predict(AZIMUTH_DEG), dtype=float)
    fine_azimuth = np.linspace(0.0, 360.0, 1441)
    curve = np.asarray(predict(fine_azimuth), dtype=float)

    if not np.all(np.isfinite(curve)) or np.any(curve <= 0.0):
        raise ValueError(f"{name}: invalid fitted directional range.")

    stats = fit_statistics(OBSERVED_RANGE_M, predicted)
    stats.update(
        {
            "Rmax_m": float(np.max(curve)),
            "Rmin_m": float(np.min(curve)),
            "Rmax_Rmin": float(np.max(curve) / np.min(curve)),
        }
    )

    return {
        "name": name,
        "predict": predict,
        "params": params,
        "stats": stats,
        "predicted": predicted,
        "residual": predicted - OBSERVED_RANGE_M,
        "fine_azimuth": fine_azimuth,
        "curve": curve,
    }


def save_outputs(models, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(
        {"Azimuth_deg": AZIMUTH_DEG, "Observed_Range_m": OBSERVED_RANGE_M}
    )
    summary_rows = []

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
        pd.DataFrame(
            {
                "Azimuth_deg": AZIMUTH_DEG,
                "Observed_Range_m": OBSERVED_RANGE_M,
                "Predicted_Range_m": model["predicted"],
                "Residual_m": model["residual"],
            }
        ).to_csv(model_dir / "observed_predicted_residual.csv", index=False)

        x, y = polar_to_xy(model["fine_azimuth"], model["curve"])
        pd.DataFrame(
            {
                "Azimuth_deg": model["fine_azimuth"],
                "Range_m": model["curve"],
                "Easting_m": x,
                "Northing_m": y,
            }
        ).to_csv(model_dir / "fitted_curve.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "nls_summary.csv", index=False)
    comparison.to_csv(output_dir / "nls_observed_vs_predicted.csv", index=False)

    # Geometry comparison.
    fine = models[0]["fine_azimuth"]
    obs_x, obs_y = polar_to_xy(AZIMUTH_DEG, OBSERVED_RANGE_M)
    limit = 1.16 * max(
        float(np.max(OBSERVED_RANGE_M)),
        *(float(np.max(model["curve"])) for model in models),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), constrained_layout=True)
    for ax, model in zip(axes, models):
        x, y = polar_to_xy(fine, model["curve"])
        ax.plot(np.r_[obs_x, obs_x[0]], np.r_[obs_y, obs_y[0]], "--", lw=1.2, label="Observed range")
        ax.scatter(obs_x, obs_y, s=28, zorder=3)
        ax.plot(x, y, lw=2.2, label="NLS fit")
        ax.set(
            title=model["name"],
            xlabel="Easting (m)",
            ylabel="Northing (m)",
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            aspect="equal",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.annotate(
            "N",
            xy=(0.08, 0.94),
            xytext=(0.08, 0.80),
            xycoords="axes fraction",
            ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>"),
        )
        s = model["stats"]
        ax.text(
            0.5,
            -0.13,
            f"RMSE = {s['RMSE_m']:.3f} m | R² = {s['R2']:.4f} | Rmax/Rmin = {s['Rmax_Rmin']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    fig.suptitle("Saprolite directional-range NLS fitting: Ellipse vs Cassini Oval", fontsize=14)
    fig.savefig(output_dir / "nls_geometry_comparison.png", dpi=250, bbox_inches="tight")

    # Range/residual comparison.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for j, model in enumerate(models):
        axes[0, j].plot(AZIMUTH_DEG, model["predicted"], label="Predicted")
        axes[0, j].scatter(AZIMUTH_DEG, OBSERVED_RANGE_M, s=25, label="Observed", zorder=3)
        axes[0, j].set_title(model["name"])
        axes[0, j].set_ylabel("Directional range (m)")
        axes[0, j].grid(alpha=0.2)
        axes[0, j].legend(fontsize=8)

        axes[1, j].axhline(0.0, lw=1.0)
        axes[1, j].vlines(AZIMUTH_DEG, 0.0, model["residual"])
        axes[1, j].scatter(AZIMUTH_DEG, model["residual"], s=25)
        axes[1, j].set_xlabel("Geological azimuth (deg)")
        axes[1, j].set_ylabel("Predicted - observed (m)")
        axes[1, j].grid(alpha=0.2)

    fig.suptitle("Saprolite NLS fit and residuals", fontsize=14)
    fig.savefig(output_dir / "nls_fit_and_residuals.png", dpi=250, bbox_inches="tight")

    return summary


def main(output_dir=DEFAULT_OUTPUT_DIR, show=True):
    output_dir = Path(output_dir)

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
    print(f"\nOutputs saved to: {output_dir.resolve()}")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary, {model["name"]: model["predict"] for model in models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLS fitting of saprolite directional ranges: Ellipse vs Cassini Oval."
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for tables and figures.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening interactive plot windows.",
    )
    args = parser.parse_args()
    main(args.out_dir, show=not args.no_show)
