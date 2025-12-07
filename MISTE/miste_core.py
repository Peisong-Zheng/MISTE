"""
miste_core.py

Core implementation of MISTE:
Multiscale Irregular-Sampling Transfer Entropy

Implements:
- Gap-based multiscale L-values
- Gaussian kernel smoothing at each scale L (window half-width ≤ L/2)
- Common grid with step ~ L
- Gap-based segmenting: only transitions where original gaps < L/2
- TE estimators at lag 1:
    * Linear-Gaussian TE (Granger-style) [optional]
    * Binned discrete TE (default; no Gaussian assumption)
- Multiscale MISTE
- MISTE-AE: TE(L) with age ensembles (COPRA-style)
- Plot helper for TE(L) with error bars
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional

# Import your ObservedSeries dataclass from syn_data.py
# from syn_data import ObservedSeries


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
    return L_values*2 # multiplied by 2 according to Nyquist criterion


# ------------------------------------------------------------
# 2. Gaussian kernel smoothing at scale L
# ------------------------------------------------------------

def smooth_gaussian_irregular(
    t_obs: np.ndarray,
    v_obs: np.ndarray,
    grid: np.ndarray,
    L: float,
    window_half_width_factor: float = 0.5,
    sigma_factor: float = 1/3,
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
    grid = np.asarray(grid, float)

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
# 3. Gap-based support mask at scale L
# ------------------------------------------------------------

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

    half_L = 0.5 * L

    for i, tau in enumerate(grid):
        # Find where tau would be inserted to keep t sorted
        idx = np.searchsorted(t, tau)
        if idx == 0 or idx == t.size:
            continue  # tau outside the bracket of observed times

        # Gap across the bracket containing tau
        gap = t[idx] - t[idx - 1]
        if gap <= half_L:
            support[i] = True

    return support


# ------------------------------------------------------------
# 4. Linear-Gaussian TE at lag 1 (optional)
# ------------------------------------------------------------

# def gaussian_te_lag1(
#     Y_future: np.ndarray,
#     Y_past: np.ndarray,
#     X_past: np.ndarray,
# ) -> float:
#     """
#     Estimate TE (bits) from X -> Y at lag 1 under a linear-Gaussian model.

#     T_{X->Y} = 0.5 * log( var(e_Y|Ypast) / var(e_Y|Ypast,Xpast) ) / log(2)

#     This is Granger causality expressed in TE units, and is valid if the
#     underlying process is linear + Gaussian.

#     Parameters
#     ----------
#     Y_future : array-like
#         Y at time n+1.
#     Y_past : array-like
#         Y at time n.
#     X_past : array-like
#         X at time n.

#     Returns
#     -------
#     TE_bits : float
#         Transfer entropy in bits. NaN if too few samples or degenerate.
#     """
#     Yf = np.asarray(Y_future, float)
#     Yp = np.asarray(Y_past, float)
#     Xp = np.asarray(X_past, float)

#     mask = np.isfinite(Yf) & np.isfinite(Yp) & np.isfinite(Xp)
#     Yf = Yf[mask]
#     Yp = Yp[mask]
#     Xp = Xp[mask]

#     N = Yf.size
#     if N <= 5:
#         return np.nan

#     # Model 1: Yf ~ a0 + a1 Yp
#     A = np.column_stack([np.ones(N), Yp])
#     beta_A, *_ = np.linalg.lstsq(A, Yf, rcond=None)
#     res_A = Yf - A @ beta_A
#     kA = A.shape[1]
#     dof_A = max(N - kA, 1)
#     var_A = np.sum(res_A**2) / dof_A

#     # Model 2: Yf ~ b0 + b1 Yp + b2 Xp
#     B = np.column_stack([np.ones(N), Yp, Xp])
#     beta_B, *_ = np.linalg.lstsq(B, Yf, rcond=None)
#     res_B = Yf - B @ beta_B
#     kB = B.shape[1]
#     dof_B = max(N - kB, 1)
#     var_B = np.sum(res_B**2) / dof_B

#     if var_A <= 0 or var_B <= 0:
#         return np.nan

#     te_nats = 0.5 * np.log(var_A / var_B)
#     te_bits = te_nats / np.log(2.0)
#     return te_bits








def gaussian_te_lag1(
    Y_future: np.ndarray,
    Y_past: np.ndarray,
    X_past: np.ndarray,
) -> float:
    """
    Estimate TE (bits) from X -> Y at lag 1 under a linear-Gaussian model.

    T_{X->Y} = 0.5 * log( SSR_A / SSR_B ) / log(2),
    where SSR_A is the residual sum of squares for Y_{t+1} ~ Y_t,
    and SSR_B is for Y_{t+1} ~ Y_t + X_t.

    This is Granger causality in TE units and is guaranteed ≥ 0 up to
    floating-point noise.
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

    # Model A: Yf ~ a0 + a1 Yp
    A = np.column_stack([np.ones(N), Yp])
    beta_A, *_ = np.linalg.lstsq(A, Yf, rcond=None)
    res_A = Yf - A @ beta_A
    SSR_A = np.sum(res_A**2)

    # Model B: Yf ~ b0 + b1 Yp + b2 Xp
    B = np.column_stack([np.ones(N), Yp, Xp])
    beta_B, *_ = np.linalg.lstsq(B, Yf, rcond=None)
    res_B = Yf - B @ beta_B
    SSR_B = np.sum(res_B**2)

    if SSR_A <= 0 or SSR_B <= 0:
        return np.nan

    # In theory SSR_B <= SSR_A, so ratio >= 1
    ratio = SSR_A / SSR_B
    if ratio <= 0:
        return np.nan

    te_nats = 0.5 * np.log(ratio)
    te_bits = te_nats / np.log(2.0)

    # Optional: clamp tiny negatives caused by numerical noise
    if te_bits < 0 and te_bits > -1e-8:
        te_bits = 0.0

    return te_bits

# ------------------------------------------------------------
# 5. Binned (discrete) TE at lag 1
# ------------------------------------------------------------

def _bin_edges_1d(values: np.ndarray, n_bins: int, method: str = "quantile") -> np.ndarray:
    """
    Compute bin edges for 1D data.

    Parameters
    ----------
    values : array-like
        Data to bin.
    n_bins : int
        Number of bins.
    method : {"quantile", "uniform"}
        Binning strategy.

    Returns
    -------
    edges : np.ndarray
        Bin edges of length n_bins + 1.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to bin.")

    if method == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(v, qs)
        # Ensure strictly increasing by adding tiny jitter if necessary
        edges = np.asarray(edges, float)
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
    elif method == "uniform":
        vmin, vmax = np.min(v), np.max(v)
        if vmax == vmin:
            vmax = vmin + 1e-9
        edges = np.linspace(vmin, vmax, n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Expand slightly to avoid points on exact boundary falling outside
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def binned_te_lag1(
    Y_future: np.ndarray,
    Y_past: np.ndarray,
    X_past: np.ndarray,
    n_bins_x: int = 6,
    n_bins_y: int = 6,
    n_bins_y_future: Optional[int] = None,
    binning: str = "quantile",
    alpha: float = 1.0,
    min_samples: int = 50,
) -> float:
    """
    Estimate TE (bits) from X -> Y at lag 1 using a discrete (binned) estimator.

    Classic TE definition:
        T_{X->Y} = sum p(y_{t+1}, y_t, x_t)
                        log [ p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) ]

    We:
      - Bin X_past, Y_past, Y_future into discrete states.
      - Build a 3D histogram with Laplace smoothing (alpha).
      - Compute TE from the smoothed probabilities.

    Parameters
    ----------
    Y_future, Y_past, X_past : array-like
        Continuous values at lag-1 transitions.
    n_bins_x, n_bins_y, n_bins_y_future : int
        Number of bins for X_t, Y_t, and Y_{t+1}. If n_bins_y_future is None,
        we use n_bins_y.
    binning : {"quantile", "uniform"}
        Binning method for all three variables.
    alpha : float
        Laplace smoothing constant added to each bin.
    min_samples : int
        Minimum number of transitions required; else return NaN.

    Returns
    -------
    TE_bits : float
        Estimated transfer entropy in bits. NaN if too few samples.
    """
    Yf = np.asarray(Y_future, float)
    Yp = np.asarray(Y_past, float)
    Xp = np.asarray(X_past, float)

    mask = np.isfinite(Yf) & np.isfinite(Yp) & np.isfinite(Xp)
    Yf = Yf[mask]
    Yp = Yp[mask]
    Xp = Xp[mask]

    N = Yf.size
    if N < min_samples:
        return np.nan

    if n_bins_y_future is None:
        n_bins_y_future = n_bins_y

    # Compute bin edges for each variable
    edges_x = _bin_edges_1d(Xp, n_bins_x, method=binning)
    edges_y = _bin_edges_1d(Yp, n_bins_y, method=binning)
    edges_yf = _bin_edges_1d(Yf, n_bins_y_future, method=binning)

    # Digitize -> bin indices 0..n_bins-1
    X_bin = np.digitize(Xp, edges_x) - 1
    Yp_bin = np.digitize(Yp, edges_y) - 1
    Yf_bin = np.digitize(Yf, edges_yf) - 1

    # Clip just in case
    X_bin = np.clip(X_bin, 0, n_bins_x - 1)
    Yp_bin = np.clip(Yp_bin, 0, n_bins_y - 1)
    Yf_bin = np.clip(Yf_bin, 0, n_bins_y_future - 1)

    # 3D histogram: counts[k, j, i] for (Yf_bin=k, Yp_bin=j, X_bin=i)
    counts = np.zeros((n_bins_y_future, n_bins_y, n_bins_x), dtype=float)
    for k, j, i in zip(Yf_bin, Yp_bin, X_bin):
        counts[k, j, i] += 1.0

    # If effectively no data, return NaN
    if counts.sum() < min_samples:
        return np.nan

    # Laplace smoothing (to avoid division by zero)
    counts_s = counts + alpha  
    total = counts_s.sum()

    # Marginals
    counts_j_i = counts_s.sum(axis=0)      # sum over k -> shape (J, I)
    counts_k_j = counts_s.sum(axis=2)      # sum over i -> shape (K, J)
    counts_j = counts_j_i.sum(axis=1)      # sum over i -> shape (J,)

    p_xyz = counts_s / total               # shape (K, J, I)
    p_j_i = counts_j_i / total             # shape (J, I)
    p_k_j = counts_k_j / total             # shape (K, J)
    p_j = counts_j / total                 # shape (J,)

    # Conditional probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        p_k_given_j_i = p_xyz / p_j_i[None, :, :]
        p_k_given_j = p_k_j / p_j[None, :]

        # Broadcast p(k|j) over the X dimension
        ratio = p_k_given_j_i / p_k_given_j[:, :, None]
        # Avoid log(0) or log of invalid numbers
        mask_pos = (p_xyz > 0) & (ratio > 0)
        te_nats = np.sum(p_xyz[mask_pos] * np.log(ratio[mask_pos]))

    te_bits = te_nats / np.log(2.0)
    return te_bits



# functions to plot alignment

import matplotlib.pyplot as plt

def _segments_from_mask(grid: np.ndarray, mask: np.ndarray):
    """Return [(start_time, end_time), ...] for contiguous True in mask."""
    grid = np.asarray(grid, float)
    mask = np.asarray(mask, bool)
    segments = []
    start_idx = None

    for i, m in enumerate(mask):
        if m and start_idx is None:
            start_idx = i
        elif (not m) and (start_idx is not None):
            segments.append((grid[start_idx], grid[i - 1]))
            start_idx = None

    if start_idx is not None:
        segments.append((grid[start_idx], grid[len(mask) - 1]))

    return segments


def _segments_from_valid_idx(grid: np.ndarray, valid_idx: np.ndarray):
    """
    valid_idx[n] means the transition from grid[n] -> grid[n+1] is valid.
    Group contiguous valid_idx into time segments.
    """
    grid = np.asarray(grid, float)
    valid_idx = np.asarray(valid_idx, int)
    if valid_idx.size == 0:
        return []

    valid_idx = np.sort(valid_idx)
    segments = []
    start = valid_idx[0]
    prev = valid_idx[0]

    for idx in valid_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            # segment is from grid[start] to grid[prev+1]
            segments.append((grid[start], grid[prev + 1]))
            start = idx
            prev = idx

    # last segment
    segments.append((grid[start], grid[prev + 1]))
    return segments

# Build a custom legend using proxy artists
import matplotlib.patches as mpatches

def plot_alignment(
    t_x: np.ndarray,
    x_vals: np.ndarray,
    t_y: np.ndarray,
    y_vals: np.ndarray,
    grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    support_x: np.ndarray,
    support_y: np.ndarray,
    support_xy: np.ndarray,
    valid_idx: np.ndarray,
    L: float,
):
    """
    Visual diagnostic plot for a single MISTE scale L.

    Top panel: X series
      - Black points: original irregular X(t_x)
      - Blue line: smoothed X on grid
      - Orange shading: times where both X and Y are 'supported' (support_xy=True)
      - Green shading: subset where valid lag-1 transitions exist (used for TE)

    Bottom panel: Y series
      - Analogous to X.

    Parameters
    ----------
    t_x, x_vals : original irregular times and values for X
    t_y, y_vals : original irregular times and values for Y
    grid : common grid used for TE
    x_grid, y_grid : smoothed series on the grid
    support_x, support_y : per-series support masks on grid
    support_xy : combined support mask (X & Y & finite smoothed values)
    valid_idx : indices n where transition grid[n] -> grid[n+1] is used for TE
    L : scale (for annotation)
    """
    t_x = np.asarray(t_x, float)
    t_y = np.asarray(t_y, float)
    grid = np.asarray(grid, float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_x, ax_y = axes

    # --- Raw and smoothed series ---
    ax_x.plot(t_x, x_vals, "k-", ms=3, alpha=0.3, label="X raw")
    ax_y.plot(t_y, y_vals, "k-", ms=3, alpha=0.3, label="Y raw")

    ax_x.plot(grid, x_grid, "r-", alpha=0.7, label="X smoothed")
    ax_y.plot(grid, y_grid, "r-", alpha=0.7, label="Y smoothed")

    # --- Segments from combined support (support_xy) ---
    seg_supported = _segments_from_mask(grid, support_xy)
    seg_valid = _segments_from_valid_idx(grid, valid_idx)

    # Orange: where both series are supported (but may be too short for TE)
    for start, end in seg_supported:
        ax_x.axvspan(start, end, color="orange", alpha=0.25)
        ax_y.axvspan(start, end, color="orange", alpha=0.25)

    # Green: where valid lag-1 transitions actually contribute to TE
    for start, end in seg_valid:
        ax_x.axvspan(start, end, color="green", alpha=0.25)
        ax_y.axvspan(start, end, color="green", alpha=0.25)

    # --- Cosmetics / labels ---
    ax_y.set_xlabel("Time")
    ax_x.set_ylabel("X")
    ax_y.set_ylabel("Y")

    ax_x.set_title(f"MISTE alignment at scale L = {L:.3g}")
    ax_x.grid(True, alpha=0.3)
    ax_y.grid(True, alpha=0.3)


    orange_patch = mpatches.Patch(color="orange", alpha=0.15, label="Supported (X & Y)")
    green_patch = mpatches.Patch(color="green", alpha=0.25, label="Used for TE (valid transitions)")

    ax_x.legend(handles=[orange_patch, green_patch], loc="upper right")
    ax_y.legend(loc="upper right")

    plt.tight_layout()
    return fig, axes


# ------------------------------------------------------------
# 6. Single-scale TE (one age model, one scale L)
# ------------------------------------------------------------

def compute_miste_single_scale(
    t_x: np.ndarray,
    x_vals: np.ndarray,
    t_y: np.ndarray,
    y_vals: np.ndarray,
    L: float,
    grid_step_factor: float = 0.5,
    window_half_width_factor: float = 0.5,
    sigma_factor: float = 1/3,
    min_valid_data_length: int = 30,
    te_method: str = "binned",
    if_plot_alignment: bool = False,
    te_kwargs: Optional[dict] = None,
) -> float:
    """
    Compute MISTE TE (X->Y) at a single scale L for one age model.

    Steps:
      1) Smooth X and Y with Gaussian kernel at scale L (window ≤ L/2).
      2) Resample both on a common grid with step ≈ grid_step_factor * L.
      3) Use gap-based masks (original gaps < L/2) to define valid segments.
      4) Collect all valid lag-1 transitions within those segments.
      5) Estimate TE using either a Gaussian (linear) or binned estimator.

    Parameters
    ----------
    t_x, x_vals : array-like
        Times and values for X.
    t_y, y_vals : array-like
        Times and values for Y.
    L : float
        Characteristic scale / smoothing length (time units).
    grid_step_factor : float
        Grid step = grid_step_factor * L (default 0.5 => L/2).
    window_half_width_factor : float
        Half-width of smoothing window, as a fraction of L (default 0.5 = L/2).
    sigma_factor : float
        Gaussian σ = window_half_width * sigma_factor (default 1/3 => 3σ = half-width).
    min_valid_transitions : int
        Minimum number of valid grid transitions required to estimate TE.
    te_method : {"binned", "gaussian"}
        Which TE estimator to use.
    te_kwargs : dict or None
        Extra keyword arguments passed to the TE estimator.

    Returns
    -------
    TE_L : float
        TE (bits) at scale L. NaN if too few transitions or degenerate.
    """
    if te_kwargs is None:
        te_kwargs = {}

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

    # 2) Construct support masks based on original gaps < L/2
    support_x = support_mask_from_gaps(t_x, grid, L)
    support_y = support_mask_from_gaps(t_y, grid, L)

    # Combined support mask where both series and smoothed values are valid
    support_xy = (
        support_x & support_y &
        np.isfinite(x_grid) & np.isfinite(y_grid)
    )

    # Valid transitions: both n and n+1 lie within supported regions
    valid_idx = np.where(support_xy[:-1] & support_xy[1:])[0]
    if valid_idx.size < min_valid_data_length:
        return np.nan

    # Optional alignment plot for debugging / inspection
    if if_plot_alignment:
        plot_alignment(
            t_x=t_x,
            x_vals=x_vals,
            t_y=t_y,
            y_vals=y_vals,
            grid=grid,
            x_grid=x_grid,
            y_grid=y_grid,
            support_x=support_x,
            support_y=support_y,
            support_xy=support_xy,
            valid_idx=valid_idx,
            L=L,
        )

    # Extract lag-1 transitions (no cross-segment jumps because of the mask)
    Y_future = y_grid[valid_idx + 1]
    Y_past = y_grid[valid_idx]
    X_past = x_grid[valid_idx]

    # Y_past   = [y_grid[n1], y_grid[n2], y_grid[n3]]
    # Y_future = [y_grid[n1+1], y_grid[n2+1], y_grid[n3+1]]
    # X_past   = [x_grid[n1], x_grid[n2], x_grid[n3]]

    # 3) Choose TE estimator
    if te_method == "gaussian":
        return gaussian_te_lag1(Y_future, Y_past, X_past)
    elif te_method == "binned":
        return binned_te_lag1(Y_future, Y_past, X_past, **te_kwargs)
    else:
        raise ValueError(f"Unknown te_method: {te_method}")


# ------------------------------------------------------------
# 7. Multiscale MISTE for a single age model
# ------------------------------------------------------------

def compute_miste_multiscale(
    t_x: np.ndarray,
    x_vals: np.ndarray,
    t_y: np.ndarray,
    y_vals: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    **single_scale_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MISTE TE(L) for multiple scales L given one age model.

    Scales L come from the quantiles of the pooled gap distribution.
    For each L, we call compute_miste_single_scale.

    Parameters
    ----------
    t_x, x_vals, t_y, y_vals : array-like
        Irregular ages and values for X and Y.
    quantiles : sequence of float
        Quantiles of pooled age gaps used to define scales L.
    single_scale_kwargs :
        Extra keyword args for compute_miste_single_scale
        (e.g., grid_step_factor, te_method, te_kwargs).

    Returns
    -------
    L_values : np.ndarray
        Scales L.
    TE_values : np.ndarray
        TE(L) in bits, one per L. May contain NaNs if too few transitions.
    """
    L_values = compute_gap_scales(t_x, t_y, quantiles=quantiles)
    TE_values = np.full(L_values.shape, np.nan, dtype=float)

    for i, L in enumerate(L_values):
        TE_values[i] = compute_miste_single_scale(
            t_x, x_vals, t_y, y_vals, L, **single_scale_kwargs
        )

    return L_values, TE_values


# ------------------------------------------------------------
# 8. MISTE-AE: multiscale TE with age ensembles (COPRA-style)
# ------------------------------------------------------------

def compute_miste_multiscale_ae(
    X_obs,  # ObservedSeries
    Y_obs,  # ObservedSeries
    quantiles: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    **single_scale_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MISTE TE(L) using COPRA-style age ensembles (MISTE-AE).

    For each scale L:
      - For each age-model ensemble m:
            - compute TE(L) for X->Y using ages from that ensemble
      - Aggregate across ensembles to get mean and std.

    Parameters
    ----------
    X_obs, Y_obs : ObservedSeries
        Observed series with age_ensembles and values.
        Assumes both have the same number of ensembles and
        attributes: t_obs, values_obs, age_ensembles.
    quantiles : sequence of float
        Quantiles of pooled gaps (from best-estimate ages) used for scales L.
    single_scale_kwargs :
        Extra args passed to compute_miste_single_scale.

    Returns
    -------
    L_values : np.ndarray
        Scales L.
    TE_mean : np.ndarray
        Mean TE(L) across ensembles.
    TE_std : np.ndarray
        Std TE(L) across ensembles.
    TE_all : np.ndarray
        Raw TE matrix of shape (n_ensembles, n_L).
    """
    # Use best-estimate ages (t_obs) to define scales
    t_x_base = X_obs.t_obs
    t_y_base = Y_obs.t_obs

    L_values = compute_gap_scales(t_x_base, t_y_base, quantiles=quantiles)

    n_ensembles = X_obs.age_ensembles.shape[0]
    if Y_obs.age_ensembles.shape[0] != n_ensembles:
        raise ValueError("X_obs and Y_obs must have the same number of ensembles.")

    n_L = L_values.size
    TE_all = np.full((n_ensembles, n_L), np.nan, dtype=float)

    for m in range(n_ensembles):
        t_x_m = X_obs.age_ensembles[m]
        t_y_m = Y_obs.age_ensembles[m]
        x_vals = X_obs.values_obs
        y_vals = Y_obs.values_obs

        for i, L in enumerate(L_values):
            TE_all[m, i] = compute_miste_single_scale(
                t_x_m, x_vals, t_y_m, y_vals, L, **single_scale_kwargs
            )

    TE_mean = np.nanmean(TE_all, axis=0)
    TE_std = np.nanstd(TE_all, axis=0, ddof=1)

    return L_values, TE_mean, TE_std, TE_all


# ------------------------------------------------------------
# 9. Plot helper – multiscale TE(L) with error bars
# ------------------------------------------------------------

def plot_miste_multiscale(
    L_values: np.ndarray,
    TE_mean: np.ndarray,
    TE_std: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    label: str = "X → Y",
    xscale_log: bool = False,
):
    """
    Plot multiscale TE(L) as points with optional error bars.

    Parameters
    ----------
    L_values : array-like
        Scales L (time units).
    TE_mean : array-like
        TE(L) mean values.
    TE_std : array-like or None
        Optional TE(L) standard deviation for error bars.
    ax : matplotlib Axes or None
        Axis to plot on. If None, a new figure and axis are created.
    label : str
        Label for the curve.
    xscale_log : bool
        If True, use log scale for the x-axis (L).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    L_values = np.asarray(L_values, float)
    TE_mean = np.asarray(TE_mean, float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if TE_std is not None:
        TE_std = np.asarray(TE_std, float)
        ax.errorbar(
            L_values,
            TE_mean,
            yerr=TE_std,
            fmt="o-",
            capsize=3,
            label=label,
        )
    else:
        ax.plot(L_values, TE_mean, "o-", label=label)

    ax.set_xlabel("Scale L (time units)")
    ax.set_ylabel("TE X → Y (bits)")
    ax.set_title("MISTE multiscale TE")
    if xscale_log:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax
