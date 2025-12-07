import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ObservedSeries:
    t_obs: np.ndarray          # true times of the observations (no error)
    values_obs: np.ndarray          # observed proxy values
    age_sigma: np.ndarray       # 1σ age uncertainty at each point
    age_ensembles: np.ndarray   # (n_ensembles, n_obs) age-model realizations



@dataclass
class MISTESyntheticData:
    t_true: np.ndarray          # regular grid for underlying processes
    X_true: np.ndarray          # true driver process on grid
    Y_true: np.ndarray          # true response process on grid
    X_components: np.ndarray    # (n_scales, n_t) X per timescale
    Y_components: np.ndarray    # (n_scales, n_t) Y per timescale
    tau_scales: np.ndarray      # (n_scales,) timescales
    coupled_scale_index: int    # which scale is actually coupled X -> Y
    X_obs: ObservedSeries       # irregularly sampled X with age ensembles
    Y_obs: ObservedSeries       # irregularly sampled Y with age ensembles


def generate_miste_synthetic_data(
    t_min=0.0,
    t_max=800.0,
    dt_true=0.1,
    tau_scales=(10.0, 1.0, 0.1),
    coupled_scale_index=1,
    coupling_strength=0.3,
    var_scales_X=(1.0, 1.0, 1.0),
    var_scales_Y=(1.0, 1.0, 1.0),
    gap_old=5.0,
    gap_young=0.1,
    n_ensembles=100,
    sigma_old=3.0,
    sigma_young=0.5,
    random_state=None,
    if_plot=False,
) -> MISTESyntheticData:
    """
    Generate synthetic data for testing MISTE (Multiscale Irregular-Sampling TE).

    - Two multiscale OU-like processes X(t), Y(t) on a fine regular grid.
    - Only ONE chosen timescale is dynamically coupled X -> Y.
    - Each series is then sampled on its own irregular time axis
      (coarser in ancient part, finer towards present).
    - Age uncertainties grow from present to ancient; COPRA-style age
      ensembles are generated for each observation.

    Time axis increases from ancient (t_min) to now (t_max).

    Returns
    -------
    MISTESyntheticData
    """
    rng = np.random.default_rng(random_state)

    # ---------------------------
    # 1) Underlying regular time grid
    # ---------------------------
    t_true = np.arange(t_min, t_max + dt_true / 2.0, dt_true)
    n_t = t_true.size

    tau_scales = np.asarray(tau_scales, dtype=float)
    var_scales_X = np.asarray(var_scales_X, dtype=float)
    var_scales_Y = np.asarray(var_scales_Y, dtype=float)
    n_scales = tau_scales.size

    assert 0 <= coupled_scale_index < n_scales, "coupled_scale_index out of range"
    assert var_scales_X.size == n_scales
    assert var_scales_Y.size == n_scales

    # Allocate per-scale arrays
    X_scales = np.zeros((n_scales, n_t))
    Y_scales = np.zeros((n_scales, n_t))

    # ---------------------------
    # 2) Simulate multiscale OU-like processes
    # ---------------------------
    for s in range(n_scales):
        tau = tau_scales[s]
        phi = np.exp(-dt_true / tau)  # OU -> AR(1)
        # Innovation variance for desired stationary variance
        sigma_eps_X = np.sqrt(var_scales_X[s] * (1.0 - phi**2))
        sigma_eps_Y = np.sqrt(var_scales_Y[s] * (1.0 - phi**2))

        # Start in approximate stationary distribution
        X_scales[s, 0] = rng.normal(scale=np.sqrt(var_scales_X[s]))
        Y_scales[s, 0] = rng.normal(scale=np.sqrt(var_scales_Y[s]))

        for k in range(1, n_t):
            # X_s: always OU
            X_scales[s, k] = (
                phi * X_scales[s, k-1] +
                rng.normal(scale=sigma_eps_X)
            )

            # Y_s: OU plus coupling from X_s at chosen scale
            if s == coupled_scale_index:
                Y_scales[s, k] = (
                    phi * Y_scales[s, k-1] +
                    coupling_strength * X_scales[s, k-1] +
                    rng.normal(scale=sigma_eps_Y)
                )
            else:
                Y_scales[s, k] = (
                    phi * Y_scales[s, k-1] +
                    rng.normal(scale=sigma_eps_Y)
                )

    X_true = X_scales.sum(axis=0)
    Y_true = Y_scales.sum(axis=0)

    # ---------------------------
    # 3) Irregular sampling (coarse ancient, fine recent)
    # ---------------------------
    def generate_irregular_times(t_min, t_max, gap_old, gap_young):
        times = []
        t = t_min
        while t < t_max:
            # 1 at t_min (ancient), 0 at t_max (present)
            frac_old = (t_max - t) / (t_max - t_min)
            mean_gap = gap_young + (gap_old - gap_young) * frac_old
            dt = rng.exponential(mean_gap)
            t = t + dt
            if t < t_max:
                times.append(t)
        return np.array(times)

    t_obs_X_true = generate_irregular_times(t_min, t_max, gap_old, gap_young)
    t_obs_Y_true = generate_irregular_times(t_min, t_max, gap_old, gap_young)

    # Use numpy.interp as the "existing module" for interpolation
    x_obs = np.interp(t_obs_X_true, t_true, X_true)
    y_obs = np.interp(t_obs_Y_true, t_true, Y_true)

    # ---------------------------
    # 4) Age uncertainties (grow toward ancient)
    # ---------------------------
    def age_sigma(t):
        # frac_old: 1 at t_min, 0 at t_max
        frac_old = (t_max - t) / (t_max - t_min)
        return sigma_young + (sigma_old - sigma_young) * frac_old

    sigma_X = age_sigma(t_obs_X_true)
    sigma_Y = age_sigma(t_obs_Y_true)

    # COPRA-style ensembles: use normal distribution from numpy
    age_X_ensembles = rng.normal(
        loc=t_obs_X_true[None, :],
        scale=sigma_X[None, :],
        size=(n_ensembles, t_obs_X_true.size),
    )
    age_Y_ensembles = rng.normal(
        loc=t_obs_Y_true[None, :],
        scale=sigma_Y[None, :],
        size=(n_ensembles, t_obs_Y_true.size),
    )

    X_obs = ObservedSeries(
        t_obs=t_obs_X_true,
        values_obs=x_obs,
        age_sigma=sigma_X,
        age_ensembles=age_X_ensembles,
    )
    Y_obs = ObservedSeries(
        t_obs=t_obs_Y_true,
        values_obs=y_obs,
        age_sigma=sigma_Y,
        age_ensembles=age_Y_ensembles,
    )

    data = MISTESyntheticData(
        t_true=t_true,
        X_true=X_true,
        Y_true=Y_true,
        X_components=X_scales,
        Y_components=Y_scales,
        tau_scales=tau_scales,
        coupled_scale_index=coupled_scale_index,
        X_obs=X_obs,
        Y_obs=Y_obs,
    )

    if if_plot:
        plot_age_uncertainty(data)
        plot_coupled_scales(data)
    return data


def plot_age_uncertainty(data: MISTESyntheticData):
    """Plot how age uncertainties grow with time for X and Y."""
    t_true = data.t_true

    # Recreate the sigma(t) curve from endpoints:
    t_min, t_max = t_true[0], t_true[-1]
    # Use observed sigmas to infer sigma_old/young for plotting
    # (or just recompute them if you have the parameters handy)
    sig_X = data.X_obs.age_sigma
    sig_Y = data.Y_obs.age_sigma

    # For a "theoretical" curve, we can fit a simple linear
    # between the mean at old and young ends of observed times.
    # Or just show observed points; here I’ll do both X & Y scatters.
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(
        data.X_obs.t_obs,
        data.X_obs.age_sigma,
        s=15,
        alpha=0.7,
        label="X age σ (obs points)",
    )
    ax.scatter(
        data.Y_obs.t_obs,
        data.Y_obs.age_sigma,
        s=15,
        alpha=0.7,
        label="Y age σ (obs points)",
        marker="x",
    )

    ax.set_xlabel("True time (ancient → present)")
    ax.set_ylabel("Age uncertainty σ (time units)")
    ax.set_title("Age model uncertainty vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_coupled_scales(data: MISTESyntheticData, downsample: int = 10):
    """
    Plot scale components of X and Y, highlighting the coupled scale.
    """
    t = data.t_true[::downsample]
    Xc = data.X_components[:, ::downsample]
    Yc = data.Y_components[:, ::downsample]
    tau = data.tau_scales
    s_c = data.coupled_scale_index
    n_scales = tau.size

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- X components ---
    ax = axes[0]
    for s in range(n_scales):
        if s == s_c:
            ax.plot(
                t, Xc[s],
                linewidth=2.5,
                label=f"X scale {s} (τ={tau[s]:.1f}, coupled)",
            )
        else:
            ax.plot(
                t, Xc[s],
                linewidth=1.0,
                alpha=0.6,
                label=f"X scale {s} (τ={tau[s]:.1f})",
            )
    # plot the x_obs
    ax.plot(
        data.X_obs.t_obs,
        data.X_obs.values_obs,
        'k.-',
        alpha=0.5,
        label="X observed",
    )
    ax.set_ylabel("X components")
    ax.set_title("Per-scale components of X(t) and Y(t)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Y components ---
    ax = axes[1]
    for s in range(n_scales):
        if s == s_c:
            ax.plot(
                t, Yc[s],
                linewidth=2.5,
                label=f"Y scale {s} (τ={tau[s]:.1f}, coupled)",
            )
        else:
            ax.plot(
                t, Yc[s],
                linewidth=1.0,
                alpha=0.6,
                label=f"Y scale {s} (τ={tau[s]:.1f})",
            )

    # plot the y_obs
    ax.plot(
        data.Y_obs.t_obs,
        data.Y_obs.values_obs,
        'k.-',
        alpha=0.5,
        label="Y observed",
    )
    ax.set_xlabel("True time (ancient → present)")
    ax.set_ylabel("Y components")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes
