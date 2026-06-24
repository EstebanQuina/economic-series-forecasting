"""
Functional Autoregressive Model FAR(1) with Haar Basis Smoothing
=================================================================
Dataset  : Latin America GDP growth rates (1961–2024)

Functional domain (FIX #1)
---------------------------
Each year's cross-country GDP growth vector is a functional observation
whose index is the COUNTRY axis, ordered by long-run mean growth rate
(not alphabetically) to give the piecewise-constant Haar basis a
meaningful monotone gradient to approximate.

Haar basis construction (FIX #2)
---------------------------------
Basis functions are stored in COLUMNS from the start and sampled on
a regular grid of length n_points.  QR re-orthonormalisation is applied
only when n_points is not a power of two.

Evaluation protocol (FIX #3)
------------------------------
Hyperparameter tuning uses an inner validation window (2015–2019).
Final evaluation uses the held-out outer window (2020–2024).
The two windows are never mixed.

Venezuela / extreme values (FIX #4)
-------------------------------------
Countries with |GDP growth| > WINSOR_LIMIT are flagged before
interpolation.  Linear interpolation is restricted to interior NaNs
(limit_area='inside') to avoid bridging structural breaks.

Operator norm check (FIX #6)
------------------------------
After each fit, ‖Ψ‖₂ (spectral norm) is computed and checked.
Stationarity requires ‖Ψ‖₂ < 1.

Prediction convention (FIX #7)
--------------------------------
Ψ satisfies  C1 ≈ C0 @ Ψ  (ridge solve).
One-step forecast: c_{t+1} = Ψᵀ c_t,  x_pred = B @ c_pred.
Docstring now matches implementation exactly.

Dead code removed (minor fix)
------------------------------
`smooth_with_haar` (never called in v1) has been removed.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from itertools import product


# ──────────────────────────────────────────────────────────────────────────────
# 1.  HAAR BASIS  (FIX #2 — basis functions stored in COLUMNS throughout)
# ──────────────────────────────────────────────────────────────────────────────

def haar_basis(n_points: int, n_basis: int) -> np.ndarray:
    """
    Build a Haar wavelet basis matrix of shape (n_points, n_basis).

    Basis functions are stored as COLUMNS (consistent with standard
    linear-algebra convention B @ c = reconstructed signal).

    When n_points is not a power of two, the full p2×p2 orthonormal
    matrix is built, then rows are sampled at equispaced grid indices,
    and QR re-orthonormalisation restores column-orthonormality.

    Parameters
    ----------
    n_points : number of evaluation points (here = n_countries)
    n_basis  : number of basis functions to retain

    Returns
    -------
    B : ndarray (n_points, n_basis), orthonormal columns
    """
    p2 = 1
    while p2 < n_points:
        p2 <<= 1

    # Build full p2×p2 Haar matrix — basis functions in COLUMNS
    H = np.zeros((p2, p2))
    H[:, 0] = 1.0 / np.sqrt(p2)          # scaling function (column 0)

    col = 1
    level_width = p2 // 2
    while level_width >= 1 and col < p2:
        for start in range(0, p2, 2 * level_width):
            if col >= p2:
                break
            psi = np.zeros(p2)
            psi[start : start + level_width] =  1.0
            psi[start + level_width : start + 2 * level_width] = -1.0
            psi /= np.sqrt(2 * level_width)
            H[:, col] = psi               # stored in COLUMN (fix from v1)
            col += 1
        level_width //= 2

    # Sample rows at n_points equispaced grid indices
    idx = np.round(np.linspace(0, p2 - 1, n_points)).astype(int)
    B_sampled = H[idx, :]                 # (n_points, p2)

    # Re-orthonormalise via QR when sampling breaks orthonormality
    if n_points != p2:
        Q, _ = np.linalg.qr(B_sampled)
        B_full = Q                        # (n_points, n_points)
    else:
        B_full = B_sampled                # already orthonormal

    n_basis = min(n_basis, B_full.shape[1])
    return B_full[:, :n_basis]            # (n_points, n_basis)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FAR(1) ESTIMATOR  (FIX #7 — consistent Ψ convention & docstring)
# ──────────────────────────────────────────────────────────────────────────────

class FAR1:
    """
    Functional Autoregressive Model of order 1.

    Coefficient-space representation
    ---------------------------------
    Each functional observation x_t ∈ R^{n_points} is encoded as

        c_t = Bᵀ x_t   ∈ R^{n_basis}

    The FAR(1) model in coefficient space is:

        c_{t+1} ≈ Ψᵀ c_t

    Ridge estimation
    ----------------
    Stack T-1 consecutive pairs into matrices:

        C0 = [c_1, …, c_{T-1}]ᵀ   shape (T-1, n_basis)   predictors
        C1 = [c_2, …, c_T  ]ᵀ   shape (T-1, n_basis)   responses

    Solve:  (C0ᵀ C0 + α I) Ψ = C0ᵀ C1

    This gives  C1 ≈ C0 Ψ,  so  c_{t+1} ≈ Ψᵀ c_t.

    Stationarity condition
    ----------------------
    ‖Ψ‖₂ < 1  (spectral / operator norm) is a sufficient condition.
    This is checked and reported after every fit call.

    Parameters
    ----------
    n_basis   : number of Haar basis functions
    reg_alpha : Tikhonov regularisation strength
    """

    def __init__(self, n_basis: int = 8, reg_alpha: float = 1e-3):
        self.n_basis        = n_basis
        self.reg_alpha      = reg_alpha
        self.Psi_           = None   # (n_basis, n_basis): C1 ≈ C0 @ Psi_
        self.B_             = None   # (n_points, n_basis)
        self.spectral_norm_ = None   # ‖Ψ‖₂

    # ── helpers ──────────────────────────────────────────────────────────────

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """
        Project rows of X (T × n_points) onto the Haar basis.
        Returns coefficient matrix C (T × n_basis).
        Builds B_ on first call; validates shape on subsequent calls.
        """
        T, n = X.shape
        if self.B_ is None:
            self.B_ = haar_basis(n, self.n_basis)
        elif n != self.B_.shape[0]:
            raise ValueError(
                f"Input has {n} points but model was fitted on "
                f"{self.B_.shape[0]} points."
            )
        return X @ self.B_           # (T, n_basis)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "FAR1":
        """
        Fit the FAR(1) model.

        Parameters
        ----------
        X : ndarray (T, n_points)
            Rows are consecutive functional observations.
            Here T = n_train_years, n_points = n_countries.
        """
        C  = self._encode(X)
        C0 = C[:-1]                      # (T-1, n_basis) predictors
        C1 = C[1:]                       # (T-1, n_basis) responses

        # Ridge solve: (C0ᵀ C0 + α I) Ψ = C0ᵀ C1
        A = C0.T @ C0 + self.reg_alpha * np.eye(self.n_basis)
        B = C0.T @ C1
        self.Psi_ = np.linalg.solve(A, B)          # (n_basis, n_basis)
        self.spectral_norm_ = np.linalg.norm(self.Psi_, ord=2)
        return self

    def predict_one_step(self, x_last: np.ndarray) -> np.ndarray:
        """
        Predict the next functional observation from the last one.

        Steps
        -----
        1. Encode:    c_last = Bᵀ x_last              (n_basis,)
        2. Propagate: c_pred = Ψᵀ c_last              (n_basis,)
           [from C1 ≈ C0 Ψ  →  c_{t+1} = Ψᵀ c_t]
        3. Decode:    x_pred = B c_pred               (n_points,)

        Parameters
        ----------
        x_last : ndarray (n_points,)

        Returns
        -------
        x_pred : ndarray (n_points,)
        """
        if self.B_ is None or self.Psi_ is None:
            raise RuntimeError("Call fit() before predict_one_step().")
        if x_last.shape[0] != self.B_.shape[0]:
            raise ValueError(
                f"x_last length {x_last.shape[0]} != model n_points "
                f"{self.B_.shape[0]}."
            )
        c_last = self.B_.T @ x_last          # (n_basis,)
        c_pred = self.Psi_.T @ c_last        # (n_basis,)  — Ψᵀ c_t
        x_pred = self.B_ @ c_pred            # (n_points,)
        return x_pred


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & PREPROCESSING  (FIX #4 — Venezuela / extreme flagging)
# ──────────────────────────────────────────────────────────────────────────────

WINSOR_LIMIT = 35.0   # pp — values beyond this are flagged before imputation

def load_data(path: str, first_year: int = 1961,
              verbose: bool = True) -> pd.DataFrame:
    """
    Load the wide-format CSV; return DataFrame (countries × years).

    Preprocessing
    -------------
    1. Flag countries with |GDP growth| > WINSOR_LIMIT (e.g. Venezuela).
    2. Apply linear interpolation only within interior NaN runs
       (limit_area='inside') to avoid bridging structural breaks.
    3. Forward-fill then back-fill remaining edge NaNs.
    """
    raw = pd.read_csv(path, index_col=0)
    raw = raw.set_index("Country")
    year_cols = [c for c in raw.columns if c.startswith("YR")]
    df = raw[year_cols].copy()
    df.columns = [int(c.replace("YR", "")) for c in df.columns]
    df = df.loc[:, df.columns >= first_year].astype(float)

    # Extreme-value flagging  (FIX #4)
    extreme_mask = df.abs() > WINSOR_LIMIT
    if verbose and extreme_mask.any().any():
        flagged = df[extreme_mask.any(axis=1)].index.tolist()
        print(
            f"\n[WARNING] Countries with |GDP growth| > {WINSOR_LIMIT} pp "
            f"(retained but flagged):\n  {flagged}"
        )
        for country in flagged:
            extremes = df.loc[country, df.loc[country].abs() > WINSOR_LIMIT]
            print(f"  {country}: {dict(extremes.round(2))}")

    # Interpolation restricted to interior gaps (FIX #4)
    df = df.T.interpolate(method="linear", limit_area="inside").T
    df = df.T.ffill().bfill().T
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROLLING-ORIGIN CV  (FIX #1 — time as functional index, ordered countries)
# ──────────────────────────────────────────────────────────────────────────────

def rolling_origin_cv(
    df: pd.DataFrame,
    cv_years: list,
    n_basis: int = 8,
    reg_alpha: float = 1e-3,
    verbose: bool = False,
) -> dict:
    """
    Rolling-origin (expanding-window) cross-validation.

    Functional observation
    ----------------------
    x_y ∈ R^{n_countries},  x_y[j] = GDP growth of country j in year y.
    The functional index (domain) is the country axis; time indexes
    the sequence of functional observations fed to FAR(1).

    Country ordering  (FIX #1 — not alphabetical)
    -----------------------------------------------
    Countries are sorted by long-run mean GDP growth so the functional
    domain has a monotone gradient that makes Haar approximation
    geometrically meaningful.  The ordering is computed on the full
    dataset (it is a structural, not predictive, attribute).

    For each horizon year h:
      • Training  : all years y < h   (expanding window)
      • Prediction: year h

    Returns
    -------
    dict {year -> {actual, predicted, countries, spectral_norm, Psi}}
    """
    # Stable ordering by long-run mean growth  (FIX #1)
    country_order = df.mean(axis=1).sort_values().index.tolist()
    df_ord   = df.loc[country_order]
    countries = country_order
    all_years = sorted(df.columns.tolist())
    results   = {}

    for h in cv_years:
        train_years = [y for y in all_years if y < h]
        if len(train_years) < 3:
            print(f"  Skipping {h}: not enough training years.")
            continue

        # X_train: (n_train_years, n_countries)  — rows = functional obs.
        X_train = df_ord[train_years].T.values.astype(float)

        model = FAR1(n_basis=n_basis, reg_alpha=reg_alpha)
        model.fit(X_train)

        sn     = model.spectral_norm_
        status = "stationary" if sn < 1 else "NON-STATIONARY"
        if verbose:
            print(f"  {h}: ‖Ψ‖₂ = {sn:.4f}  [{status}]")

        x_last   = X_train[-1]
        x_pred   = model.predict_one_step(x_last)
        x_actual = df_ord[h].values.astype(float)

        results[h] = {
            "actual":        x_actual,
            "predicted":     x_pred,
            "countries":     countries,
            "spectral_norm": sn,
            "Psi":           model.Psi_.copy(),
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: dict):
    countries = results[list(results.keys())[0]]["countries"]
    all_act, all_pred, yearly_rows = [], [], []

    for year, res in sorted(results.items()):
        act, pred = res["actual"], res["predicted"]
        all_act.append(act); all_pred.append(pred)
        yearly_rows.append({
            "Year": year,
            "RMSE": np.sqrt(((act - pred) ** 2).mean()),
            "MAE":  np.abs(act - pred).mean(),
            "spectral_norm": res["spectral_norm"],
        })

    all_act  = np.array(all_act)
    all_pred = np.array(all_pred)

    per_country_rows = []
    for i, c in enumerate(countries):
        sq = (all_act[:, i] - all_pred[:, i]) ** 2
        ab = np.abs(all_act[:, i] - all_pred[:, i])
        per_country_rows.append({
            "Country": c,
            "RMSE":    np.sqrt(sq.mean()),
            "MAE":     ab.mean(),
        })

    per_country = pd.DataFrame(per_country_rows).sort_values("RMSE")
    yearly      = pd.DataFrame(yearly_rows)
    overall     = {
        "RMSE": np.sqrt(((all_act - all_pred) ** 2).mean()),
        "MAE":  np.abs(all_act - all_pred).mean(),
    }
    return per_country, overall, yearly


# ──────────────────────────────────────────────────────────────────────────────
# 6.  GRID SEARCH — inner window only  (FIX #3)
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(df, inner_years, basis_options, alpha_options):
    """
    Select hyperparameters using ONLY the inner validation window.
    The outer (final evaluation) window is never touched here.
    """
    best_rmse, best_params, records = np.inf, None, []

    for nb, ra in product(basis_options, alpha_options):
        res = rolling_origin_cv(df, inner_years, n_basis=nb,
                                reg_alpha=ra, verbose=False)
        if not res:
            continue
        _, overall, _ = compute_metrics(res)
        records.append({"n_basis": nb, "reg_alpha": ra, **overall})
        if overall["RMSE"] < best_rmse:
            best_rmse, best_params = overall["RMSE"], (nb, ra)

    return best_params, pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

LATAM_BLUE  = "#003087"
LATAM_RED   = "#CE1126"
LATAM_GOLD  = "#F4A900"
BG_COLOR    = "#F8F9FA"
PANEL_COLOR = "#FFFFFF"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_COLOR)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#222222", pad=6)
    ax.set_xlabel(xlabel, fontsize=9, color="#555555")
    ax.set_ylabel(ylabel, fontsize=9, color="#555555")
    ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")


def plot_results(df, results, per_country, overall, yearly,
                 n_basis, reg_alpha, save_path="far1_haar_results.png"):
    cv_years  = sorted(results.keys())
    countries = results[cv_years[0]]["countries"]
    n_c       = len(countries)
    all_years = sorted(df.columns.tolist())

    fig = plt.figure(figsize=(22, 28), facecolor=BG_COLOR)
    fig.suptitle(
        "FAR(1) with Haar Basis — Latin America GDP Growth\n"
        f"Inner tuning 2015–2019  |  Outer evaluation 2020–2024  |  "
        f"n_basis={n_basis}, α={reg_alpha}  |  "
        f"Overall RMSE={overall['RMSE']:.4f} pp,  MAE={overall['MAE']:.4f} pp",
        fontsize=13, fontweight="bold", color="#111111", y=0.995,
    )
    gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.55, wspace=0.40,
                           left=0.06, right=0.97, top=0.96, bottom=0.03)

    # (A) Historical trajectories
    ax_hist = fig.add_subplot(gs[0, :2])
    cmap = plt.cm.tab20
    for i, ctry in enumerate(countries):
        ax_hist.plot(all_years, df.loc[ctry, all_years],
                     color=cmap(i / n_c), lw=1.0, alpha=0.75, label=ctry)
    ax_hist.axvspan(2015, 2019, color="grey",     alpha=0.12, label="Inner CV")
    ax_hist.axvspan(2019, 2024, color=LATAM_GOLD, alpha=0.15, label="Outer eval")
    _style_ax(ax_hist, "Historical GDP Growth (1961–2024)", "Year", "Growth (%)")
    ax_hist.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.5)

    # (B) Haar basis functions
    ax_haar = fig.add_subplot(gs[0, 2:])
    B = haar_basis(n_c, n_basis)
    t = np.linspace(0, 1, n_c)
    for j in range(min(n_basis, 8)):
        ax_haar.plot(t, B[:, j], color=plt.cm.cool(j / n_basis),
                     lw=1.5, label=f"φ_{j}")
    _style_ax(ax_haar,
              f"Haar Basis (first {min(n_basis,8)} of {n_basis})\n"
              "[domain = countries ordered by mean GDP growth]",
              "Country rank (normalised)", "Amplitude")
    ax_haar.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.5)

    # (C) Actual vs Predicted per year
    for k, year in enumerate(cv_years):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        act, pred = results[year]["actual"], results[year]["predicted"]
        lim = max(np.abs(act).max(), np.abs(pred).max()) * 1.15
        ax.scatter(act, pred, color=LATAM_BLUE, edgecolors="white",
                   s=50, alpha=0.85, zorder=3)
        ax.plot([-lim, lim], [-lim, lim], color=LATAM_RED,
                lw=1.2, ls="--", label="45°")
        rmse_y = np.sqrt(((act - pred) ** 2).mean())
        mae_y  = np.abs(act - pred).mean()
        sn     = results[year]["spectral_norm"]
        ax.set_title(f"{year}  RMSE={rmse_y:.2f}  MAE={mae_y:.2f}  ‖Ψ‖₂={sn:.3f}",
                     fontsize=8, fontweight="bold", color="#222222")
        ax.set_xlabel("Actual (%)", fontsize=8)
        ax.set_ylabel("Predicted (%)", fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_facecolor(PANEL_COLOR)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, ls="--", alpha=0.35, color="#CCCCCC")
        ax.tick_params(labelsize=8)

    # (D) Spectral norm over years  (FIX #6)
    ax_sn = fig.add_subplot(gs[2, :2])
    sn_vals = [results[y]["spectral_norm"] for y in cv_years]
    ax_sn.plot(cv_years, sn_vals, "o-", color=LATAM_BLUE, lw=2, ms=8)
    ax_sn.axhline(1.0, color=LATAM_RED, lw=1.5, ls="--",
                  label="Stationarity boundary (‖Ψ‖₂=1)")
    for y, v in zip(cv_years, sn_vals):
        ax_sn.annotate(f"{v:.3f}", (y, v),
                       textcoords="offset points", xytext=(0, 9),
                       fontsize=8, color=LATAM_BLUE, ha="center")
    _style_ax(ax_sn, "Operator Norm ‖Ψ‖₂ by CV Year (stationarity check)",
              "Year", "‖Ψ‖₂")
    ax_sn.legend(fontsize=9)
    ax_sn.set_xticks(cv_years)

    # (E) Per-country RMSE
    ax_bar = fig.add_subplot(gs[3, :2])
    pc_s = per_country.sort_values("RMSE", ascending=True)
    colors_bar = [LATAM_BLUE if r < pc_s["RMSE"].median() else LATAM_RED
                  for r in pc_s["RMSE"]]
    bars = ax_bar.barh(pc_s["Country"], pc_s["RMSE"],
                       color=colors_bar, edgecolor="white", height=0.7)
    ax_bar.axvline(overall["RMSE"], color=LATAM_GOLD, lw=2, ls="--",
                   label=f"Overall RMSE={overall['RMSE']:.3f}")
    for bar, v in zip(bars, pc_s["RMSE"]):
        ax_bar.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left", fontsize=7.5)
    _style_ax(ax_bar, "Per-Country RMSE (Outer CV 2020–2024)", "RMSE (pp)", "")
    ax_bar.legend(fontsize=8)

    # (F) Per-country MAE
    ax_mae = fig.add_subplot(gs[3, 2:])
    pc_m = per_country.sort_values("MAE", ascending=True)
    colors_mae = [LATAM_BLUE if r < pc_m["MAE"].median() else LATAM_RED
                  for r in pc_m["MAE"]]
    bars2 = ax_mae.barh(pc_m["Country"], pc_m["MAE"],
                        color=colors_mae, edgecolor="white", height=0.7)
    ax_mae.axvline(overall["MAE"], color=LATAM_GOLD, lw=2, ls="--",
                   label=f"Overall MAE={overall['MAE']:.3f}")
    for bar, v in zip(bars2, pc_m["MAE"]):
        ax_mae.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left", fontsize=7.5)
    _style_ax(ax_mae, "Per-Country MAE (Outer CV 2020–2024)", "MAE (pp)", "")
    ax_mae.legend(fontsize=8)

    # (G) Yearly RMSE & MAE
    ax_yr = fig.add_subplot(gs[4, :2])
    ax_yr.plot(yearly["Year"], yearly["RMSE"], "o-", color=LATAM_BLUE,
               lw=2, ms=7, label="RMSE")
    ax_yr.plot(yearly["Year"], yearly["MAE"], "s--", color=LATAM_RED,
               lw=2, ms=7, label="MAE")
    for _, r in yearly.iterrows():
        ax_yr.annotate(f"{r['RMSE']:.2f}", (r["Year"], r["RMSE"]),
                       textcoords="offset points", xytext=(0, 8),
                       fontsize=8, color=LATAM_BLUE, ha="center")
        ax_yr.annotate(f"{r['MAE']:.2f}", (r["Year"], r["MAE"]),
                       textcoords="offset points", xytext=(0, -12),
                       fontsize=8, color=LATAM_RED, ha="center")
    _style_ax(ax_yr, "RMSE & MAE by Outer CV Year", "Year", "Error (pp)")
    ax_yr.legend(fontsize=9)
    ax_yr.set_xticks(cv_years)

    # (H) Error heatmap
    ax_heat = fig.add_subplot(gs[4, 2:])
    err_mat = np.array([results[y]["actual"] - results[y]["predicted"]
                        for y in cv_years])
    vmax = np.abs(err_mat).max()
    cmap_heat = LinearSegmentedColormap.from_list(
        "rw", [LATAM_RED, "white", LATAM_BLUE], N=256)
    im = ax_heat.imshow(err_mat, aspect="auto", cmap=cmap_heat,
                        vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(range(len(cv_years)))
    ax_heat.set_yticklabels(cv_years, fontsize=8)
    ax_heat.set_xticks(range(n_c))
    ax_heat.set_xticklabels(countries, rotation=60, ha="right", fontsize=7)
    ax_heat.set_title(
        "Prediction Error Heatmap (Actual − Predicted, pp)\n"
        "[countries ordered by mean GDP growth]",
        fontsize=10, fontweight="bold", color="#222222")
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label="Error (pp)")

    # (I) Note on grid search
    ax_gs = fig.add_subplot(gs[5, :])
    ax_gs.axis("off")
    ax_gs.text(0.5, 0.5,
               "Hyperparameters selected on inner CV 2015–2019 only. "
               "Full grid-search results in far1_grid_search.csv",
               ha="center", va="center", fontsize=10, color="#555555",
               style="italic")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Figure saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  FAR(1) Haar Basis — Latin America GDP Growth  [corrected v2]")
    print("=" * 72)

    DATA_PATH = "latin_america_gdp_growth_clean.csv"
    df = load_data(DATA_PATH, first_year=1961, verbose=True)
    print(f"\nData: {df.shape[0]} countries × {df.shape[1]} years "
          f"({df.columns.min()}–{df.columns.max()})")
    country_rank = df.mean(axis=1).sort_values().index.tolist()
    print("Countries (low→high mean growth):", ", ".join(country_rank))

    # Disjoint windows  (FIX #3)
    INNER_YEARS   = [2015, 2016, 2017, 2018, 2019]   # hyperparameter tuning
    OUTER_YEARS   = [2020, 2021, 2022, 2023, 2024]   # final evaluation
    BASIS_OPTIONS = [4, 8, 12, 16]
    ALPHA_OPTIONS = [1e-4, 1e-3, 1e-2, 0.1]

    print(f"\n── Grid search on INNER window {INNER_YEARS} ──────────────────")
    best_params, grid_df = grid_search(
        df, INNER_YEARS, BASIS_OPTIONS, ALPHA_OPTIONS)
    if best_params is None:
        raise RuntimeError("Grid search failed.")
    best_nb, best_ra = best_params
    print(f"  Best n_basis={best_nb},  reg_alpha={best_ra}")
    print(grid_df.sort_values("RMSE").to_string(index=False))

    print(f"\n── Outer evaluation {OUTER_YEARS} ──────────────────────────────")
    results = rolling_origin_cv(df, OUTER_YEARS, n_basis=best_nb,
                                reg_alpha=best_ra, verbose=True)
    per_country, overall, yearly = compute_metrics(results)

    print(f"\n  Overall  RMSE={overall['RMSE']:.4f} pp   "
          f"MAE={overall['MAE']:.4f} pp")

    # Stationarity summary  (FIX #6)
    non_stat = yearly[yearly["spectral_norm"] >= 1.0]
    if non_stat.empty:
        print("  [OK] ‖Ψ‖₂ < 1 in all outer CV folds → FAR(1) is stationary.")
    else:
        print(f"  [!] ‖Ψ‖₂ ≥ 1 in years {non_stat['Year'].tolist()} "
              f"— interpret with caution.")

    print("\n── Yearly Metrics ──────────────────────────────────────────────────")
    print(yearly.to_string(index=False))

    print("\n── Per-Country Metrics (sorted by RMSE) ────────────────────────────")
    print(per_country.to_string(index=False))

    print("\n── Actual vs Predicted ─────────────────────────────────────────────")
    for year in sorted(results.keys()):
        act, pred, ctrs = (results[year]["actual"], results[year]["predicted"],
                           results[year]["countries"])
        sn = results[year]["spectral_norm"]
        detail = pd.DataFrame({
            "Country":   ctrs,
            "Actual":    np.round(act, 3),
            "Predicted": np.round(pred, 3),
            "Error":     np.round(act - pred, 3),
        })
        print(f"\n  Year {year}  ‖Ψ‖₂={sn:.4f}:")
        print(detail.to_string(index=False))

    plot_results(df, results, per_country, overall, yearly,
                 n_basis=best_nb, reg_alpha=best_ra)

    per_country.to_csv("far1_per_country_metrics.csv", index=False)
    yearly.to_csv("far1_yearly_metrics.csv", index=False)
    grid_df.sort_values("RMSE").to_csv("far1_grid_search.csv", index=False)
    print("\nCSV summaries saved. Done.")


if __name__ == "__main__":
    main()
