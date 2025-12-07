"""
miste_core.py

Core implementation of MISTE:
Multiscale Irregular-Sampling Transfer Entropy

Implements:
- Gap-based multiscale L-values
- Spline smoothing at each scale L
- Common grid with step ~L
- Gap-based segmenting: only transitions where original gaps < L
- Linear-Gaussian TE at lag 1
- MISTE-AE: TE(L) with age ensembles (COPRA-style)
- Plot helper for TE(L) with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional

from dataclasses import dataclass
# from syn_data import ObservedSeries, MISTESyntheticData
# import syn_data as sd

# If SciPy is available, use UnivariateSpline for smoothing.
# If not, you can later swap this for another smoother.
from scipy.interpolate import UnivariateSpline





# ------------------------------------------------------------
# 1. Gap-based scales L from age gaps
# ------------------------------------------------------------

def compute_gap_scales(
    t_x: np.ndarray,
    t_y: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    min_gap: Optional[float] = None,
    max_gap: Optional[float] = None,
) -> np.ndarray:
    """
    Compute characteristic scales L from the empirical distribution of age gaps.

    Parameters
    ----------
    t_x, t_y : array-like
        Age arrays for X and Y (best-estimate ages, or one ensemble).
    quantiles : sequence of float
        Quantiles of the pooled gap distribution used as scales L.
    min_gap, max_gap : float or None
        Optional filters on the gap distribution.

    Returns
    -------
    L_values : np.ndarray
        Array of scales L (same units as t_x/t_y).
    """
    t_x = np.asarray(t_x, float)
    t_y = np.asarray(t_y, float)

    dx = np.diff(np.sort(t_x))
    dy = np.diff(np.sort(t_y))
    gaps = np.concatenate([dx, dy])
    gaps = gaps[gaps > 0]

    if min_gap is not None:
        gaps = gaps[gaps >= min_gap]
    if max_gap is not None:
        gaps = gaps[gaps <= max_gap]

    if gaps.size == 0:
        raise ValueError("No positive gaps available to compute scales.")

    L_values = np.quantile(gaps, quantiles)
    return L_values





# ------------------------------------------------------------
# 2. Smoothing at scale L to remove processes with time scales < L
# to safely preserve only processes with time scales ≥ L, the smoothing lenth should be <=L/2 according to the Nyquist criterion.
# ------------------------------------------------------------
import numpy as np

def smooth_gaussian_irregular(
    t_obs: np.ndarray,
    v_obs: np.ndarray,
    grid: Optional[np.ndarray] = None,
    L: float = 1.0,
    window_half_width_factor: float = 0.5,
    sigma_factor: float = 1/3,   # 3σ = window_half_width
    weight_threshold: float = 1e-6,
) -> np.ndarray:
    """
    Gaussian kernel smoothing for irregularly sampled data.

    For each grid time τ, we:
      - Only use observations with |t_obs - τ| <= window_half_width_factor * L
      - Within that window, apply Gaussian weights with
            σ = (window_half_width_factor * L) * sigma_factor
        By default, 3σ = L/2 so the effective window half-width is L/2.

    This implements the idea that the smoothing window should be ≤ L/2,
    so we preserve L-scale variability reasonably well while strongly
    damping shorter scales.

    Parameters
    ----------
    t_obs, v_obs : array-like
        Irregular observation times and values.
    grid : array-like
        Grid times at which to evaluate the smoothed series.
    L : float
        Target timescale (time units).
    window_half_width_factor : float
        Half-width = window_half_width_factor * L (default 0.5 => L/2).
    sigma_factor : float
        Gaussian σ = window_half_width * sigma_factor (default 1/3 => 3σ = half-width).
    weight_threshold : float
        Minimum total weight required to accept a smoothed value.

    Returns
    -------
    vals : np.ndarray
        Smoothed values at each grid point (NaN where not enough support).
    """
    t = np.asarray(t_obs, float)
    v = np.asarray(v_obs, float)
    # grid = np.asarray(grid, float)

    if grid is None:
        grid = np.arange(np.min(t), np.max(t), L/2)

    # sort observations by time
    order = np.argsort(t)
    t = t[order]
    v = v[order]

    vals = np.full(grid.shape, np.nan, dtype=float)

    if t.size == 0:
        return vals

    half_width = window_half_width_factor * L
    sigma = half_width * sigma_factor  # default: sigma = (L/2) / 3 = L/6

    for i, tau in enumerate(grid):
        # restrict to points within half-width window
        mask = np.abs(t - tau) <= half_width
        if not np.any(mask):
            continue

        d = t[mask] - tau
        w = np.exp(-0.5 * (d / sigma) ** 2)
        W = w.sum()
        if W < weight_threshold:
            continue

        vals[i] = np.dot(w, v[mask]) / W

    return vals













# ------------------------------------------------------------
# 0. Linear-Gaussian TE at lag 1
# ------------------------------------------------------------

def gaussian_te_lag1(
    Y_future: np.ndarray,
    Y_past: np.ndarray,
    X_past: np.ndarray,
) -> float:
    """
    Estimate TE (bits) from X -> Y at lag 1 under a linear-Gaussian model.

    T_{X->Y} = 0.5 * log( var(e_Y|Ypast) / var(e_Y|Ypast,Xpast) ) / log(2)

    Parameters
    ----------
    Y_future : array-like
        Y at time n+1.
    Y_past : array-like
        Y at time n.
    X_past : array-like
        X at time n.

    Returns
    -------
    TE_bits : float
        Transfer entropy in bits. NaN if too few samples or degenerate.
    """
    Yf = np.asarray(Y_future, float)
    Yp = np.asarray(Y_past, float)
    Xp = np.asarray(X_past, float)

    mask = np.isfinite(Yf) & np.isfinite(Yp) & np.isfinite(Xp)
    Yf = Yf[mask]
    Yp = Yp[mask]
    Xp = Xp[mask]

    N = Yf.size
    if N <= 5:
        return np.nan

    # Model 1: Yf ~ a0 + a1 Yp
    A = np.column_stack([np.ones(N), Yp])
    beta_A, *_ = np.linalg.lstsq(A, Yf, rcond=None)
    res_A = Yf - A @ beta_A
    kA = A.shape[1]
    dof_A = max(N - kA, 1)
    var_A = np.sum(res_A**2) / dof_A

    # Model 2: Yf ~ b0 + b1 Yp + b2 Xp
    B = np.column_stack([np.ones(N), Yp, Xp])
    beta_B, *_ = np.linalg.lstsq(B, Yf, rcond=None)
    res_B = Yf - B @ beta_B
    kB = B.shape[1]
    dof_B = max(N - kB, 1)
    var_B = np.sum(res_B**2) / dof_B

    if var_A <= 0 or var_B <= 0:
        return np.nan

    te_nats = 0.5 * np.log(var_A / var_B)
    te_bits = te_nats / np.log(2.0)
    return te_bits







# # ------------------------------------------------------------
# # 3. Gap-based support mask at scale L
# # ------------------------------------------------------------

def support_mask_from_gaps(
    t_obs: np.ndarray,
    grid: np.ndarray,
    L: float,
) -> np.ndarray:
    """
    For each grid point, decide if it's within a segment where original
    data gaps are < L/2.

    For a grid time tau:
        - Find i such that t_obs[i-1] <= tau <= t_obs[i]
        - If such bracket exists and gap = t_obs[i] - t_obs[i-1] <= L/2,
          then tau is considered supported.

    Parameters
    ----------
    t_obs : array-like
        Irregular observation times (1D, not necessarily sorted).
    grid : array-like
        Grid times where we have smoothed values.
    L : float
        Smoothing length / resolution scale.

    Returns
    -------
    support : np.ndarray (bool)
        Support mask for each grid time.
    """
    t = np.sort(np.asarray(t_obs, float))
    grid = np.asarray(grid, float)

    support = np.zeros(grid.shape, dtype=bool)

    if t.size < 2:
        return support  # no gaps to define

    for i, tau in enumerate(grid):
        # Find where tau would be inserted to keep t sorted
        idx = np.searchsorted(t, tau)
        if idx == 0 or idx == t.size:
            continue  # tau outside the bracket of observed times

        gap = t[idx] - t[idx - 1]
        if gap <= L/2:
            support[i] = True

    return support


# # ------------------------------------------------------------
# # 4. Single-scale MISTE (one age model, one scale L)
# # ------------------------------------------------------------

from typing import Sequence, Tuple
import numpy as np



def compute_miste_single_scale(
    t_x: np.ndarray,
    x_vals: np.ndarray,
    t_y: np.ndarray,
    y_vals: np.ndarray,
    L: float,
    grid_step_factor: float = 0.5,
    window_half_width_factor: float = 0.5,
    sigma_factor: float = 1/3,
    min_valid_transitions: int = 30,
) -> float:
    """
    Compute MISTE TE (X->Y) at a single scale L for one age model,
    using Gaussian kernel smoothing with window half-width ≤ L/2.

    Steps:
      1) Smooth X and Y with Gaussian kernel at scale L (window ≤ L/2).
      2) Resample both on a common grid with step ≈ L.
      3) Use gap-based masks (original gaps < L) to define valid segments.
      4) Compute TE via linear-Gaussian model on all valid lag-1 transitions.

    Parameters
    ----------
    t_x, x_vals : array-like
        Times and values for X.
    t_y, y_vals : array-like
        Times and values for Y.
    L : float
        Characteristic scale / smoothing length (time units).
    grid_step_factor : float
        Grid step = grid_step_factor * L.
    window_half_width_factor : float
        Half-width of smoothing window, as a fraction of L (default 0.5 = L/2).
    sigma_factor : float
        Gaussian σ = window_half_width * sigma_factor (default 1/3 => 3σ = half-width).
    min_valid_transitions : int
        Minimum number of valid grid transitions required to estimate TE.

    Returns
    -------
    TE_L : float
        TE (bits) at scale L. NaN if too few transitions or degenerate.
    """
    t_x = np.asarray(t_x, float)
    t_y = np.asarray(t_y, float)
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)

    # Overlapping domain in time
    t_min = max(t_x.min(), t_y.min())
    t_max = min(t_x.max(), t_y.max())

    if t_max - t_min < 2 * L:
        return np.nan

    dt_grid = grid_step_factor * L
    grid = np.arange(t_min, t_max + dt_grid / 2.0, dt_grid)
    if grid.size < 3:
        return np.nan

    # 1) Smooth each series at scale L via Gaussian kernel (window ≤ L/2)
    x_grid = smooth_gaussian_irregular(
        t_x, x_vals, grid, L,
        window_half_width_factor=window_half_width_factor,
        sigma_factor=sigma_factor,
    )
    y_grid = smooth_gaussian_irregular(
        t_y, y_vals, grid, L,
        window_half_width_factor=window_half_width_factor,
        sigma_factor=sigma_factor,
    )

    # 2) Construct support masks based on original gaps < L
    support_x = support_mask_from_gaps(t_x, grid, L)
    support_y = support_mask_from_gaps(t_y, grid, L)

    # Valid transitions: both series supported at n and n+1, and smoothed values not NaN
    valid_idx = []
    for n in range(grid.size - 1):
        if (support_x[n] and support_x[n + 1] and
            support_y[n] and support_y[n + 1] and
            np.isfinite(x_grid[n]) and np.isfinite(x_grid[n+1]) and
            np.isfinite(y_grid[n]) and np.isfinite(y_grid[n+1])):
            valid_idx.append(n)
    valid_idx = np.asarray(valid_idx, int)

    if valid_idx.size < min_valid_transitions:
        return np.nan

    # 3) TE at lag 1 over these valid transitions
    Y_future = y_grid[valid_idx + 1]
    Y_past = y_grid[valid_idx]
    X_past = x_grid[valid_idx]

    te_L = gaussian_te_lag1(Y_future, Y_past, X_past)
    return te_L


# # ------------------------------------------------------------
# # 5. Multiscale MISTE for a single age model
# # ------------------------------------------------------------

# def compute_miste_multiscale(
#     t_x: np.ndarray,
#     x_vals: np.ndarray,
#     t_y: np.ndarray,
#     y_vals: np.ndarray,
#     quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#     **single_scale_kwargs,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute MISTE TE(L) for multiple scales L given one age model.

#     Scales L come from the quantiles of the pooled gap distribution.
#     For each L, we call compute_miste_single_scale.

#     Parameters
#     ----------
#     t_x, x_vals, t_y, y_vals : array-like
#         Irregular ages and values for X and Y.
#     quantiles : sequence of float
#         Quantiles of pooled age gaps used to define scales L.
#     single_scale_kwargs :
#         Extra keyword args for compute_miste_single_scale
#         (e.g., grid_step_factor, s_factor_x, s_factor_y).

#     Returns
#     -------
#     L_values : np.ndarray
#         Scales L.
#     TE_values : np.ndarray
#         TE(L) in bits, one per L. May contain NaNs if too few transitions.
#     """
#     L_values = compute_gap_scales(t_x, t_y, quantiles=quantiles)
#     TE_values = np.full(L_values.shape, np.nan, dtype=float)

#     for i, L in enumerate(L_values):
#         TE_values[i] = compute_miste_single_scale(
#             t_x, x_vals, t_y, y_vals, L, **single_scale_kwargs
#         )

#     return L_values, TE_values


# # ------------------------------------------------------------
# # 6. MISTE-AE: multiscale TE with age ensembles (COPRA-style)
# # ------------------------------------------------------------

# def compute_miste_multiscale_ae(
#     X_obs: ObservedSeries,
#     Y_obs: ObservedSeries,
#     quantiles: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
#     **single_scale_kwargs,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute MISTE TE(L) using COPRA-style age ensembles (MISTE-AE).

#     For each scale L:
#       - For each age-model ensemble m:
#             - compute TE(L) for X->Y using ages from that ensemble
#       - Aggregate across ensembles to get mean and std.

#     Parameters
#     ----------
#     X_obs, Y_obs : ObservedSeries
#         Observed series with age_ensembles and values.
#         Assumes both have the same number of ensembles.
#     quantiles : sequence of float
#         Quantiles of pooled gaps (from best-estimate ages) used for scales L.
#     single_scale_kwargs :
#         Extra args passed to compute_miste_single_scale.

#     Returns
#     -------
#     L_values : np.ndarray
#         Scales L.
#     TE_mean : np.ndarray
#         Mean TE(L) across ensembles.
#     TE_std : np.ndarray
#         Std TE(L) across ensembles.
#     TE_all : np.ndarray
#         Raw TE matrix of shape (n_ensembles, n_L).
#     """
#     # Use best-estimate ages (t_true) to define scales
#     t_x_base = X_obs.t_true
#     t_y_base = Y_obs.t_true

#     L_values = compute_gap_scales(t_x_base, t_y_base, quantiles=quantiles)

#     n_ensembles = X_obs.age_ensembles.shape[0]
#     assert Y_obs.age_ensembles.shape[0] == n_ensembles, "X/Y must have same n_ensembles"

#     n_L = L_values.size
#     TE_all = np.full((n_ensembles, n_L), np.nan, dtype=float)

#     for m in range(n_ensembles):
#         t_x_m = X_obs.age_ensembles[m]
#         t_y_m = Y_obs.age_ensembles[m]
#         x_vals = X_obs.values
#         y_vals = Y_obs.values

#         for i, L in enumerate(L_values):
#             TE_all[m, i] = compute_miste_single_scale(
#                 t_x_m, x_vals, t_y_m, y_vals, L, **single_scale_kwargs
#             )

#     TE_mean = np.nanmean(TE_all, axis=0)
#     TE_std = np.nanstd(TE_all, axis=0, ddof=1)

#     return L_values, TE_mean, TE_std, TE_all


# # ------------------------------------------------------------
# # 7. Plot helper – multiscale TE(L) with error bars
# # ------------------------------------------------------------

# def plot_miste_multiscale(
#     L_values: np.ndarray,
#     TE_mean: np.ndarray,
#     TE_std: Optional[np.ndarray] = None,
#     ax: Optional[plt.Axes] = None,
#     label: str = "X → Y",
#     xscale_log: bool = False,
# ):
#     """
#     Plot multiscale TE(L) as points with optional error bars.

#     Parameters
#     ----------
#     L_values : array-like
#         Scales L (time units).
#     TE_mean : array-like
#         TE(L) mean values.
#     TE_std : array-like or None
#         Optional TE(L) standard deviation for error bars.
#     ax : matplotlib Axes or None
#         Axis to plot on. If None, a new figure and axis are created.
#     label : str
#         Label for the curve.
#     xscale_log : bool
#         If True, use log scale for the x-axis (L).

#     Returns
#     -------
#     fig, ax : matplotlib Figure and Axes
#     """
#     L_values = np.asarray(L_values, float)
#     TE_mean = np.asarray(TE_mean, float)

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(5, 4))
#     else:
#         fig = ax.figure

#     if TE_std is not None:
#         TE_std = np.asarray(TE_std, float)
#         ax.errorbar(
#             L_values,
#             TE_mean,
#             yerr=TE_std,
#             fmt="o-",
#             capsize=3,
#             label=label,
#         )
#     else:
#         ax.plot(L_values, TE_mean, "o-", label=label)

#     ax.set_xlabel("Scale L (time units)")
#     ax.set_ylabel("TE X → Y (bits)")
#     ax.set_title("MISTE multiscale TE")
#     if xscale_log:
#         ax.set_xscale("log")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     plt.tight_layout()
#     return fig, ax
