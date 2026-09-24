# -*- coding: utf-8 -*-
"""NLS fitting of limonite directional ranges: Ellipse vs Cassini Oval.

This public-repository script reproduces the nonlinear least-squares (NLS)
fitting used to compare two anisotropy geometries:
    1. Elliptical anisotropy
    2. Affine-scaled Cassini Oval anisotropy

The directional ranges are embedded in this script and do not contain
confidential drillhole coordinates or assay data.

Geological azimuth convention:
    0° = North, 90° = East, clockwise.

Run from a terminal:
    python src/nls_fitting_limonite.py --no-show

Dependencies:
    numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nls_limonite"

# -----------------------------------------------------------------------------
# Directional-range observations used in the study
# -----------------------------------------------------------------------------
AZIMUTH_DEG = np.array(
    [
        0.0, 22.5, 45.0, 67.5,
        90.0, 112.5, 135.0, 157.5,
        180.0, 202.5, 225.0, 247.5,
        270.0, 292.5, 315.0, 337.5,
    ],
    dtype=float,
)

OBSERVED_RANGE_M = np.array(
    [
        155.0, 200.0, 245.0, 300.0,
        290.0, 235.0, 175.0, 153.0,
        155.0, 200.0, 245.0, 300.0,
        290.0, 235.0, 175.0, 153.0,
    ],
    dtype=float,
)


# -----------------------------------------------------------------------------
# Coordinate and statistical helpers
# -----------------------------------------------------------------------------
def geological_to_math_angle(azimuth_deg):
    """Convert geological azimuth to mathematical polar angle in radians."""
    return np.deg2rad((90.0 - np.asarray(azimuth_deg, dtype=float)) % 360.0)


def math_axis_to_geological_azimuth(phi_rad):
    """Convert a mathematical axis angle to geological azimuth modulo 180°."""
    return float((90.0 - np.degrees(phi_rad)) % 180.0)


def polar_to_east_north(azimuth_deg, radius):
    """Convert geological azimuth and radius to Easting/Northing coordinates."""
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    radius = np.asarray(radius, dtype=float)
    return radius * np.sin(az), radius * np.cos(az)


def fit_statistics(observed, predicted, n_parameters):
    """Return SSE, RMSE, MAE, R² and descriptive adjusted R²."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed

    sse = float(np.sum(residual**2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    n = len(observed)
    adjusted_r2 = (
        float(1.0 - (1.0 - r2) * (n - 1.0) / (n - n_parameters))
        if np.isfinite(r2) and n > n_parameters
        else np.nan
    )

    return {
        "SSE_m2": sse,
        "RMSE_m": rmse,
        "MAE_m": mae,
        "R2": r2,
        "Adjusted_R2_descriptive": adjusted_r2,
    }


def radial_extrema(radius_function):
    """Find global radial maximum/minimum and their azimuths on [0, 180)."""
    az = np.linspace(0.0, 180.0, 7200, endpoint=False)
    radii = np.asarray(radius_function(az), dtype=float)
    step = az[1] - az[0]

    if np.ptp(radii) <= 1e-8 * max(1.0, np.max(radii)):
        value = float(np.max(radii))
        return value, [], value, []

    extrema = []
    for sign in (-1, 1):
        score = sign * radii
        indices = np.flatnonzero(
            (score <= np.roll(score, 1)) & (score <= np.roll(score, -1))
        )
        candidates = []
        for i in indices:
            result = minimize_scalar(
                lambda angle: float(sign * radius_function(angle % 180.0)),
                bounds=(az[i] - step, az[i] + step),
                method="bounded",
                options={"xatol": 1e-10},
            )
            angle = float(result.x % 180.0)
            candidates.append((float(radius_function(angle)), angle))

        best_value = min(candidates, key=lambda item: sign * item[0])[0]
        best_angles = sorted(
            angle
            for value, angle in candidates
            if np.isclose(value, best_value, rtol=1e-7, atol=1e-7)
        )
        extrema.extend([best_value, best_angles])

    return tuple(extrema)


# -----------------------------------------------------------------------------
# Ellipse model
# -----------------------------------------------------------------------------
def ellipse_radius(theta_rad, semi_major, semi_minor, phi_rad):
    """Polar radius of an ellipse for mathematical angle theta."""
    relative_angle = theta_rad - phi_rad
    return 1.0 / np.sqrt(
        np.cos(relative_angle) ** 2 / semi_major**2
        + np.sin(relative_angle) ** 2 / semi_minor**2
    )


def fit_ellipse(azimuth_deg, observed_range_m):
    """Fit ellipse parameters by multistart nonlinear least squares."""
    theta_obs = geological_to_math_angle(azimuth_deg)

    def residual(parameters):
        log_major, log_minor, phi = parameters
        major = np.exp(log_major)
        minor = np.exp(log_minor)
        return ellipse_radius(theta_obs, major, minor, phi) - observed_range_m

    best = None
    best_sse = np.inf

    for major0 in np.linspace(220.0, 340.0, 7):
        for minor0 in np.linspace(120.0, 220.0, 6):
            for phi0_deg in np.arange(0.0, 180.0, 10.0):
                initial = np.array(
                    [np.log(major0), np.log(minor0), np.deg2rad(phi0_deg)]
                )
                result = least_squares(
                    residual,
                    initial,
                    bounds=([-10.0, -10.0, -np.pi], [10.0, 10.0, np.pi]),
                    loss="linear",
                    max_nfev=20000,
                )
                sse = float(np.sum(result.fun**2))
                if result.success and np.isfinite(sse) and sse < best_sse:
                    best = result
                    best_sse = sse

    if best is None:
        raise RuntimeError("Ellipse fitting did not converge.")

    major = float(np.exp(best.x[0]))
    minor = float(np.exp(best.x[1]))
    phi = float(best.x[2] % np.pi)

    if minor > major:
        major, minor = minor, major
        phi = float((phi + np.pi / 2.0) % np.pi)

    def predict(azimuth):
        return ellipse_radius(geological_to_math_angle(azimuth), major, minor, phi)

    params = {
        "Major_m": major,
        "Minor_m": minor,
        "Major_Minor_Ratio": major / minor,
        "Major_Axis_Azimuth_deg": math_axis_to_geological_azimuth(phi),
    }
    return predict, params


# -----------------------------------------------------------------------------
# Affine-scaled Cassini Oval model
# -----------------------------------------------------------------------------
def cassini_radius(theta_rad, phi_rad, a, c_over_a, delta):
    """Polar radius of the centered affine-scaled Cassini Oval."""
    c = c_over_a * a
    sx = np.exp(delta)
    sy = np.exp(-delta)

    relative_angle = theta_rad - phi_rad
    qx = np.cos(relative_angle) / sx
    qy = np.sin(relative_angle) / sy
    qsum = qx**2 + qy**2

    bcoef = 2.0 * a**2 * qsum - 4.0 * a**2 * qx**2
    ccoef = a**4 - c**4
    discriminant = np.maximum(bcoef**2 - 4.0 * qsum**2 * ccoef, 0.0)

    radius_squared = (-bcoef + np.sqrt(discriminant)) / (2.0 * qsum**2)
    return np.sqrt(np.maximum(radius_squared, 0.0))


def fit_cassini(azimuth_deg, observed_range_m):
    """Fit affine-scaled Cassini Oval parameters by multistart NLS."""
    theta_obs = geological_to_math_angle(azimuth_deg)

    def residual(parameters):
        phi, log_a, log_ratio_minus_one, delta = parameters
        a = np.exp(log_a)
        c_over_a = 1.0 + np.exp(log_ratio_minus_one)
        return (
            cassini_radius(theta_obs, phi, a, c_over_a, delta)
            - observed_range_m
        )

    best = None
    best_sse = np.inf

    for phi0_deg in np.arange(0.0, 180.0, 10.0):
        for a0 in [80.0, 120.0, 160.0, 200.0, 240.0]:
            for ratio0 in [1.05, 1.20, 1.50, 2.00, 3.00]:
                for delta0 in [-0.50, -0.25, 0.0, 0.25, 0.50]:
                    initial = np.array(
                        [
                            np.deg2rad(phi0_deg),
                            np.log(a0),
                            np.log(ratio0 - 1.0),
                            delta0,
                        ]
                    )
                    result = least_squares(
                        residual,
                        initial,
                        bounds=(
                            [-np.pi, np.log(1e-4), np.log(1e-6), -1.5],
                            [np.pi, np.log(3000.0), np.log(20.0), 1.5],
                        ),
                        loss="linear",
                        max_nfev=30000,
                    )
                    sse = float(np.sum(result.fun**2))
                    if result.success and np.isfinite(sse) and sse < best_sse:
                        best = result
                        best_sse = sse

    if best is None:
        raise RuntimeError("Cassini Oval fitting did not converge.")

    phi = float(best.x[0] % np.pi)
    a = float(np.exp(best.x[1]))
    c_over_a = float(1.0 + np.exp(best.x[2]))
    c = float(c_over_a * a)
    delta = float(best.x[3])
    sx = float(np.exp(delta))
    sy = float(np.exp(-delta))

    def predict(azimuth):
        return cassini_radius(
            geological_to_math_angle(azimuth), phi, a, c_over_a, delta
        )

    params = {
        "Cassini_a_m": a,
        "Cassini_c_m": c,
        "c_over_a": c_over_a,
        "sx": sx,
        "sy": sy,
        "sx_over_sy": sx / sy,
        "Focus_Axis_Azimuth_deg": math_axis_to_geological_azimuth(phi),
    }
    return predict, params


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def build_model_result(name, predict, params, observed_azimuth, observed_range):
    predicted = np.asarray(predict(observed_azimuth), dtype=float)
    stats = fit_statistics(observed_range, predicted, n_parameters=len(params))
    rmax, azmax, rmin, azmin = radial_extrema(predict)

    stats.update(
        {
            "Rmax_m": rmax,
            "Rmin_m": rmin,
            "Rmax_Rmin": rmax / rmin,
            "Rmax_Azimuth_deg": "; ".join(f"{v:.3f}" for v in azmax),
            "Rmin_Azimuth_deg": "; ".join(f"{v:.3f}" for v in azmin),
        }
    )

    return {
        "name": name,
        "predict": predict,
        "params": params,
        "stats": stats,
        "predicted": predicted,
        "residual": predicted - observed_range,
    }


def save_outputs(models, output_dir):
    """Write tables and publication-friendly diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fine_azimuth = np.linspace(0.0, 360.0, 1441)

    summary_rows = []
    comparison = pd.DataFrame(
        {"Azimuth_deg": AZIMUTH_DEG, "Observed_Range_m": OBSERVED_RANGE_M}
    )

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

        curve = np.asarray(model["predict"](fine_azimuth), dtype=float)
        east, north = polar_to_east_north(fine_azimuth, curve)
        pd.DataFrame(
            {
                "Azimuth_deg": fine_azimuth,
                "Range_m": curve,
                "Easting_m": east,
                "Northing_m": north,
            }
        ).to_csv(model_dir / "fitted_curve.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "nls_summary.csv", index=False)
    comparison.to_csv(output_dir / "nls_observed_vs_predicted.csv", index=False)

    # Polar-geometry comparison in Cartesian Easting/Northing coordinates.
    limit = 1.12 * max(
        float(np.max(OBSERVED_RANGE_M)),
        *(float(np.max(m["predict"](fine_azimuth))) for m in models),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), constrained_layout=True)
    obs_e, obs_n = polar_to_east_north(AZIMUTH_DEG, OBSERVED_RANGE_M)

    for ax, model in zip(axes, models):
        curve = model["predict"](fine_azimuth)
        east, north = polar_to_east_north(fine_azimuth, curve)
        ax.plot(np.r_[obs_e, obs_e[0]], np.r_[obs_n, obs_n[0]], "--", lw=1.2, label="Observed range")
        ax.scatter(obs_e, obs_n, s=28, zorder=3)
        ax.plot(east, north, lw=2.2, label="NLS fit")
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

    fig.suptitle("Limonite directional-range NLS fitting: Ellipse vs Cassini Oval", fontsize=14)
    fig.savefig(output_dir / "nls_geometry_comparison.png", dpi=250, bbox_inches="tight")

    # Observed/predicted and residual comparison.
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

    fig.suptitle("Limonite NLS fit and residuals", fontsize=14)
    fig.savefig(output_dir / "nls_fit_and_residuals.png", dpi=250, bbox_inches="tight")

    return summary, (fig,)


def main(output_dir=DEFAULT_OUTPUT_DIR, show=True):
    """Fit both anisotropy models and save reproducible outputs."""
    output_dir = Path(output_dir)

    print("Fitting 1/2: Ellipse ...", flush=True)
    ellipse_predict, ellipse_params = fit_ellipse(AZIMUTH_DEG, OBSERVED_RANGE_M)

    print("Fitting 2/2: Cassini Oval ...", flush=True)
    cassini_predict, cassini_params = fit_cassini(AZIMUTH_DEG, OBSERVED_RANGE_M)

    models = [
        build_model_result(
            "Ellipse", ellipse_predict, ellipse_params, AZIMUTH_DEG, OBSERVED_RANGE_M
        ),
        build_model_result(
            "Cassini_Oval", cassini_predict, cassini_params, AZIMUTH_DEG, OBSERVED_RANGE_M
        ),
    ]

    summary, _ = save_outputs(models, output_dir)

    print("\nNLS FITTING SUMMARY")
    print(summary[["Model", "RMSE_m", "MAE_m", "R2", "Rmax_Rmin"]].round(5).to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary, {m["name"]: m["predict"] for m in models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLS fitting of limonite directional ranges: Ellipse vs Cassini Oval."
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
