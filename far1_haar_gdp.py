"""
Functional Autoregressive Model FAR(1) with Haar Basis Smoothing
=================================================================
Dataset  : Latin America GDP growth rates (1961-2024)
Model    : FAR(1)  — each country's annual GDP curve is treated as a
           functional observation indexed by a discrete time grid.
Basis    : Haar wavelets (orthonormal, piecewise-constant, multi-resolution)
CV       : Rolling-origin (expanding window) on 2020-2024
Metrics  : RMSE, MAE per country and overall
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from itertools import product

# ──────────────────────────────────────────────────────────────────────────────
# 1.  HAAR BASIS
# ──────────────────────────────────────────────────────────────────────────────

def haar_basis(n_points: int, n_basis: int) -> np.ndarray:
    """
    Build a Haar wavelet basis matrix of shape (n_points, n_basis).

    The columns are the standard Haar functions evaluated on an equispaced
    grid in [0, 1):
      - col 0  : scaling function φ(t) = 1 / sqrt(n_points)
      - col k  : ψ_{j,k}(t) at resolution j and translation k

    Parameters
    ----------
    n_points : length of the discrete time grid (observations per function)
    n_basis  : number of basis functions to retain (must be a power of 2,
               or we use the first n_basis columns of the full matrix)

    Returns
    -------
    B : ndarray of shape (n_points, n_basis), orthonormal columns
    """
    # Build full Haar matrix for the next power-of-2 >= n_points
    p2 = 1
    while p2 < n_points:
        p2 <<= 1

    H = np.zeros((p2, p2))
    H[0, :] = 1.0 / np.sqrt(p2)   # scaling function

    col = 1
    level_width = p2 // 2
    while level_width >= 1 and col < p2:
        for start in range(0, p2, 2 * level_width):
            if col >= p2:
                break
            psi = np.zeros(p2)
            psi[start : start + level_width] =  1.0
            psi[start + level_width : start + 2 * level_width] = -1.0
            psi /= np.sqrt(2 * level_width)   # L2-normalise
            H[col] = psi
            col += 1
        level_width //= 2

    # Truncate rows to actual n_points (sample the basis at equispaced grid)
    idx = np.round(np.linspace(0, p2 - 1, n_points)).astype(int)
    B_full = H[:, idx].T          # shape: (n_points, p2)

    # Orthonormality is only guaranteed for the full p2×p2 case;
    # re-orthonormalise after row-truncation via QR
    Q, _ = np.linalg.qr(B_full)  # shape: (n_points, n_points)

    n_basis = min(n_basis, Q.shape[1])
    return Q[:, :n_basis]         # (n_points, n_basis)


def smooth_with_haar(y: np.ndarray, n_basis: int) -> np.ndarray:
    """
    Project a signal y onto the Haar basis and reconstruct.

    Returns the smoothed signal (same length as y) and the coefficient vector.
    """
    n = len(y)
    B = haar_basis(n, n_basis)           # (n, n_basis)
    # Least-squares projection: c = B^T y  (columns are orthonormal)
    c = B.T @ y                          # (n_basis,)
    y_smooth = B @ c                     # (n,)
    return y_smooth, c, B


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FAR(1) ESTIMATOR
# ──────────────────────────────────────────────────────────────────────────────

class FAR1:
    """
    Functional Autoregressive Model of order 1.

    Model:  X_t(s) = ∫ β(s,t) X_{t-1}(t) dt + ε_t(s)

    In the Haar coefficient space this reduces to a standard vector
    autoregression:   c_t ≈ Ψ c_{t-1}

    where Ψ is estimated by multivariate OLS (one regression per basis
    coefficient of the response):

        Ψ = C_1 C_0^+        (C_1 = lagged response matrix,
                               C_0 = predictor matrix,
                               ^+ = pseudoinverse)

    Parameters
    ----------
    n_basis   : number of Haar basis functions used per functional observation
    reg_alpha : Tikhonov regularisation strength (ridge penalty on Ψ)
    """

    def __init__(self, n_basis: int = 8, reg_alpha: float = 1e-3):
        self.n_basis   = n_basis
        self.reg_alpha = reg_alpha
        self.Psi_      = None    # (n_basis, n_basis) transition matrix
        self.B_        = None    # Haar basis matrix (n_points, n_basis)

    # ── private helpers ──────────────────────────────────────────────────────

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """
        Smooth each row of X (shape: T×n_points) and return coefficient
        matrix C (shape: T×n_basis).
        """
        T, n = X.shape
        if self.B_ is None:
            self.B_ = haar_basis(n, self.n_basis)
        C = X @ self.B_          # (T, n_basis)  — projection
        return C

    def _decode(self, C: np.ndarray) -> np.ndarray:
        """Reconstruct functional observations from coefficients."""
        return C @ self.B_.T     # (T, n_points)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray):
        """
        Fit the FAR(1) model.

        Parameters
        ----------
        X : ndarray of shape (T, n_points)
            Functional time series — rows are consecutive observations.
        """
        C = self._encode(X)           # (T, n_basis)
        C0 = C[:-1]                   # predictors  (T-1, n_basis)
        C1 = C[1:]                    # responses   (T-1, n_basis)

        # Ridge regression: Ψ = (C0^T C0 + αI)^{-1} C0^T C1
        A = C0.T @ C0 + self.reg_alpha * np.eye(self.n_basis)
        B = C0.T @ C1
        self.Psi_ = np.linalg.solve(A, B)   # (n_basis, n_basis)
        return self

    def predict_one_step(self, x_last: np.ndarray) -> np.ndarray:
        """
        Predict the next functional observation given the last one.

        Parameters
        ----------
        x_last : ndarray of shape (n_points,)

        Returns
        -------
        x_pred : ndarray of shape (n_points,)
        """
        if self.B_ is None:
            raise RuntimeError("Model not fitted yet.")
        c_last = self.B_.T @ x_last          # (n_basis,)
        c_pred = self.Psi_.T @ c_last        # (n_basis,)  — note transpose
        x_pred = self.B_ @ c_pred           # (n_points,)
        return x_pred


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str, first_year: int = 1961):
    """
    Load the wide-format CSV and return a tidy DataFrame indexed by country,
    columns = years.
    """
    raw = pd.read_csv(path, index_col=0)
    raw = raw.set_index("Country")
    year_cols = [c for c in raw.columns if c.startswith("YR")]
    df = raw[year_cols].copy()
    df.columns = [int(c.replace("YR", "")) for c in df.columns]
    df = df.loc[:, df.columns >= first_year]
    # Linear interpolation for internal NaNs; forward/back fill for edges
    df = df.T.interpolate(method="linear").ffill().bfill().T
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ROLLING-ORIGIN CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def rolling_origin_cv(
    df: pd.DataFrame,
    cv_years: list,
    n_basis: int = 8,
    reg_alpha: float = 1e-3,
):
    """
    Rolling-origin cross-validation.

    For each horizon year h in cv_years:
      • Training data  : all years up to (h-1)   [expanding window]
      • Prediction     : year h
      • The functional observation for year y is the GDP growth trajectory of
        all countries for that year, treated as a multivariate signal over
        the country dimension (index space = countries).

    Because GDP data are scalar per country per year, we treat the
    *cross-country vector* at each year as a single functional observation
    of length n_countries.  The FAR(1) model then forecasts the next
    year's cross-country profile from the current one.

    Returns
    -------
    results : dict  {year -> {'actual': array, 'predicted': array,
                              'countries': list}}
    """
    countries = df.index.tolist()
    all_years  = sorted(df.columns.tolist())
    results    = {}

    for h in cv_years:
        train_years = [y for y in all_years if y < h]
        if len(train_years) < 3:
            print(f"  Skipping {h}: not enough training years.")
            continue

        # Build training matrix: shape (n_train_years, n_countries)
        X_train = df[train_years].T.values.astype(float)   # (T, n_countries)

        model = FAR1(n_basis=n_basis, reg_alpha=reg_alpha)
        model.fit(X_train)

        # Last training observation → predict year h
        x_last = X_train[-1]                                # (n_countries,)
        x_pred = model.predict_one_step(x_last)             # (n_countries,)
        x_actual = df[h].values.astype(float)               # (n_countries,)

        results[h] = {
            "actual":    x_actual,
            "predicted": x_pred,
            "countries": countries,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: dict):
    """
    Compute RMSE and MAE per country and overall.

    Returns
    -------
    per_country : DataFrame  columns = [Country, RMSE, MAE]
    overall     : dict       {RMSE: float, MAE: float}
    yearly      : DataFrame  columns = [Year, RMSE, MAE]
    """
    countries = results[list(results.keys())[0]]["countries"]
    n_countries = len(countries)

    all_actual    = []
    all_predicted = []
    yearly_rows   = []

    for year, res in sorted(results.items()):
        act  = res["actual"]
        pred = res["predicted"]
        all_actual.append(act)
        all_predicted.append(pred)

        sq_err  = (act - pred) ** 2
        abs_err = np.abs(act - pred)
        yearly_rows.append({
            "Year": year,
            "RMSE": np.sqrt(sq_err.mean()),
            "MAE":  abs_err.mean(),
        })

    all_actual    = np.array(all_actual)    # (n_years, n_countries)
    all_predicted = np.array(all_predicted)

    # Per-country metrics (over CV years)
    per_country_rows = []
    for i, c in enumerate(countries):
        sq  = (all_actual[:, i] - all_predicted[:, i]) ** 2
        ab  = np.abs(all_actual[:, i] - all_predicted[:, i])
        per_country_rows.append({
            "Country": c,
            "RMSE":    np.sqrt(sq.mean()),
            "MAE":     ab.mean(),
        })

    per_country = pd.DataFrame(per_country_rows).sort_values("RMSE")
    yearly      = pd.DataFrame(yearly_rows)
    overall     = {
        "RMSE": np.sqrt(((all_actual - all_predicted) ** 2).mean()),
        "MAE":  np.abs(all_actual - all_predicted).mean(),
    }

    return per_country, overall, yearly


# ──────────────────────────────────────────────────────────────────────────────
# 6.  HYPERPARAMETER SEARCH  (optional grid search over n_basis & reg_alpha)
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(df, cv_years, basis_options, alpha_options):
    """Return best (n_basis, reg_alpha) minimising overall RMSE."""
    best_rmse = np.inf
    best_params = None
    records = []

    for nb, ra in product(basis_options, alpha_options):
        res = rolling_origin_cv(df, cv_years, n_basis=nb, reg_alpha=ra)
        _, overall, _ = compute_metrics(res)
        records.append({"n_basis": nb, "reg_alpha": ra, **overall})
        if overall["RMSE"] < best_rmse:
            best_rmse = overall["RMSE"]
            best_params = (nb, ra)

    grid_df = pd.DataFrame(records)
    return best_params, grid_df


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

LATAM_BLUE   = "#003087"
LATAM_RED    = "#CE1126"
LATAM_GOLD   = "#F4A900"
BG_COLOR     = "#F8F9FA"
PANEL_COLOR  = "#FFFFFF"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#222222", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#555555")
    ax.set_ylabel(ylabel, fontsize=9, color="#555555")
    ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")


def plot_results(
    df: pd.DataFrame,
    results: dict,
    per_country: pd.DataFrame,
    overall: dict,
    yearly: pd.DataFrame,
    n_basis: int,
    reg_alpha: float,
    save_path: str = "far1_haar_results.png",
):
    cv_years   = sorted(results.keys())
    countries  = results[cv_years[0]]["countries"]
    n_c        = len(countries)

    fig = plt.figure(figsize=(22, 26), facecolor=BG_COLOR)
    fig.suptitle(
        "FAR(1) with Haar Basis — Latin America GDP Growth\n"
        f"Rolling-Origin CV 2020-2024  |  n_basis={n_basis}, α={reg_alpha}  |  "
        f"Overall RMSE={overall['RMSE']:.4f} pp,  MAE={overall['MAE']:.4f} pp",
        fontsize=14, fontweight="bold", color="#111111", y=0.99
    )

    gs = gridspec.GridSpec(
        5, 4,
        figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.06, right=0.97, top=0.95, bottom=0.04
    )

    # ── (A) Historical GDP trajectories ──────────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, :2])
    cmap = plt.cm.tab20
    all_years = sorted(df.columns.tolist())
    for i, ctry in enumerate(countries):
        ax_hist.plot(all_years, df.loc[ctry, all_years],
                     color=cmap(i / n_c), lw=1.0, alpha=0.75, label=ctry)
    ax_hist.axvspan(2020, 2024, color=LATAM_GOLD, alpha=0.15, label="CV window")
    _style_ax(ax_hist, "Historical GDP Growth Trajectories (1961–2024)",
              "Year", "GDP Growth Rate (%)")
    ax_hist.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.5)

    # ── (B) Haar basis functions ─────────────────────────────────────────────
    ax_haar = fig.add_subplot(gs[0, 2:])
    t = np.linspace(0, 1, n_c)
    B = haar_basis(n_c, n_basis)
    haar_cmap = plt.cm.cool
    for j in range(min(n_basis, 8)):
        ax_haar.plot(t, B[:, j], color=haar_cmap(j / n_basis),
                     lw=1.5, label=f"φ_{j}")
    _style_ax(ax_haar, f"Haar Basis Functions (first {min(n_basis,8)} of {n_basis})",
              "Domain [0,1]", "Amplitude")
    ax_haar.legend(fontsize=7, ncol=4, loc="upper right", framealpha=0.5)

    # ── (C) Actual vs Predicted scatter per CV year ───────────────────────────
    for k, year in enumerate(cv_years):
        row = 1 + k // 4
        col = k % 4
        ax = fig.add_subplot(gs[row, col])
        act  = results[year]["actual"]
        pred = results[year]["predicted"]
        lim  = max(np.abs(act).max(), np.abs(pred).max()) * 1.15
        ax.scatter(act, pred, color=LATAM_BLUE, edgecolors="white",
                   s=50, alpha=0.85, zorder=3)
        ax.plot([-lim, lim], [-lim, lim], color=LATAM_RED,
                lw=1.2, ls="--", label="45° line")
        rmse_y = np.sqrt(((act - pred) ** 2).mean())
        mae_y  = np.abs(act - pred).mean()
        ax.set_title(f"{year}  RMSE={rmse_y:.2f}  MAE={mae_y:.2f}",
                     fontsize=9, fontweight="bold", color="#222222")
        ax.set_xlabel("Actual (%)", fontsize=8)
        ax.set_ylabel("Predicted (%)", fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_facecolor(PANEL_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, ls="--", alpha=0.35, color="#CCCCCC")
        ax.tick_params(labelsize=8)

    # ── (D) Per-country RMSE bar chart ───────────────────────────────────────
    ax_bar = fig.add_subplot(gs[3, :2])
    pc_sorted = per_country.sort_values("RMSE", ascending=True)
    colors_bar = [LATAM_BLUE if r < pc_sorted["RMSE"].median()
                  else LATAM_RED for r in pc_sorted["RMSE"]]
    bars = ax_bar.barh(pc_sorted["Country"], pc_sorted["RMSE"],
                       color=colors_bar, edgecolor="white", height=0.7)
    ax_bar.axvline(overall["RMSE"], color=LATAM_GOLD, lw=2,
                   ls="--", label=f"Overall RMSE={overall['RMSE']:.3f}")
    for bar, v in zip(bars, pc_sorted["RMSE"]):
        ax_bar.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left", fontsize=7.5)
    _style_ax(ax_bar, "Per-Country RMSE (CV 2020-2024)", "RMSE (pp)", "")
    ax_bar.legend(fontsize=8)

    # ── (E) Per-country MAE bar chart ────────────────────────────────────────
    ax_mae = fig.add_subplot(gs[3, 2:])
    pc_mae = per_country.sort_values("MAE", ascending=True)
    colors_mae = [LATAM_BLUE if r < pc_mae["MAE"].median()
                  else LATAM_RED for r in pc_mae["MAE"]]
    bars2 = ax_mae.barh(pc_mae["Country"], pc_mae["MAE"],
                        color=colors_mae, edgecolor="white", height=0.7)
    ax_mae.axvline(overall["MAE"], color=LATAM_GOLD, lw=2,
                   ls="--", label=f"Overall MAE={overall['MAE']:.3f}")
    for bar, v in zip(bars2, pc_mae["MAE"]):
        ax_mae.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", ha="left", fontsize=7.5)
    _style_ax(ax_mae, "Per-Country MAE (CV 2020-2024)", "MAE (pp)", "")
    ax_mae.legend(fontsize=8)

    # ── (F) Yearly RMSE & MAE line ────────────────────────────────────────────
    ax_yr = fig.add_subplot(gs[4, :2])
    ax_yr.plot(yearly["Year"], yearly["RMSE"], "o-", color=LATAM_BLUE,
               lw=2, ms=7, label="RMSE")
    ax_yr.plot(yearly["Year"], yearly["MAE"],  "s--", color=LATAM_RED,
               lw=2, ms=7, label="MAE")
    for _, row_ in yearly.iterrows():
        ax_yr.annotate(f"{row_['RMSE']:.2f}",
                       (row_["Year"], row_["RMSE"]),
                       textcoords="offset points", xytext=(0, 8),
                       fontsize=8, color=LATAM_BLUE, ha="center")
        ax_yr.annotate(f"{row_['MAE']:.2f}",
                       (row_["Year"], row_["MAE"]),
                       textcoords="offset points", xytext=(0, -12),
                       fontsize=8, color=LATAM_RED, ha="center")
    _style_ax(ax_yr, "RMSE & MAE by CV Horizon Year", "Year", "Error (pp)")
    ax_yr.legend(fontsize=9)
    ax_yr.set_xticks(cv_years)

    # ── (G) Prediction error heatmap ─────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[4, 2:])
    err_mat = np.array([
        results[y]["actual"] - results[y]["predicted"]
        for y in cv_years
    ])                                      # (n_cv_years, n_countries)
    vmax = np.abs(err_mat).max()
    cmap_heat = LinearSegmentedColormap.from_list(
        "rw", [LATAM_RED, "white", LATAM_BLUE], N=256
    )
    im = ax_heat.imshow(err_mat, aspect="auto", cmap=cmap_heat,
                        vmin=-vmax, vmax=vmax)
    ax_heat.set_yticks(range(len(cv_years)))
    ax_heat.set_yticklabels(cv_years, fontsize=8)
    ax_heat.set_xticks(range(n_c))
    ax_heat.set_xticklabels(countries, rotation=60, ha="right", fontsize=7)
    ax_heat.set_title("Prediction Error Heatmap (Actual − Predicted, pp)",
                      fontsize=10, fontweight="bold", color="#222222")
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label="Error (pp)")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"\nFigure saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  FAR(1) with Haar Basis — Latin America GDP Growth")
    print("=" * 68)

    # ── Load data ────────────────────────────────────────────────────────────
    DATA_PATH = "latin_america_gdp_growth_clean.csv"
    df = load_data(DATA_PATH, first_year=1961)
    print(f"\nData loaded: {df.shape[0]} countries × {df.shape[1]} years "
          f"({df.columns.min()}–{df.columns.max()})")
    print("Countries:", ", ".join(df.index.tolist()))

    # ── Hyperparameter grid search ───────────────────────────────────────────
    CV_YEARS      = [2020, 2021, 2022, 2023, 2024]
    BASIS_OPTIONS = [4, 8, 12, 16]
    ALPHA_OPTIONS = [1e-4, 1e-3, 1e-2, 0.1]

    print("\nRunning hyperparameter grid search …")
    best_params, grid_df = grid_search(
        df, CV_YEARS, BASIS_OPTIONS, ALPHA_OPTIONS
    )
    if best_params is None:
        raise RuntimeError(
            "Grid search did not find any valid hyperparameters. "
            "Check the CV years and data coverage."
        )
    best_nb, best_ra = best_params
    print(f"  Best n_basis = {best_nb},  best reg_alpha = {best_ra}")
    print("\nGrid search results (sorted by RMSE):")
    print(grid_df.sort_values("RMSE").to_string(index=False))

    # ── Final model with best params ─────────────────────────────────────────
    print(f"\nFitting FAR(1) with n_basis={best_nb}, reg_alpha={best_ra} …")
    results = rolling_origin_cv(df, CV_YEARS, n_basis=best_nb, reg_alpha=best_ra)

    per_country, overall, yearly = compute_metrics(results)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print(f"  OVERALL  RMSE = {overall['RMSE']:.4f} pp   "
          f"MAE = {overall['MAE']:.4f} pp")
    print("─" * 68)

    print("\n── Yearly Metrics ─────────────────────────────────────────────")
    print(yearly.to_string(index=False))

    print("\n── Per-Country Metrics (sorted by RMSE) ───────────────────────")
    print(per_country.to_string(index=False))

    print("\n── Actual vs Predicted (pp) ────────────────────────────────────")
    for year in sorted(results.keys()):
        print(f"\n  Year {year}:")
        act  = results[year]["actual"]
        pred = results[year]["predicted"]
        ctrs = results[year]["countries"]
        detail = pd.DataFrame({
            "Country":   ctrs,
            "Actual":    np.round(act, 3),
            "Predicted": np.round(pred, 3),
            "Error":     np.round(act - pred, 3),
        })
        print(detail.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_results(
        df, results, per_country, overall, yearly,
        n_basis=best_nb, reg_alpha=best_ra,
    )

    # ── Save CSV summaries ───────────────────────────────────────────────────
    per_country.to_csv(
        "far1_per_country_metrics.csv", index=False
    )
    yearly.to_csv(
        "far1_yearly_metrics.csv", index=False
    )
    grid_df.sort_values("RMSE").to_csv(
        "far1_grid_search.csv", index=False
    )
    print("\nCSV summaries saved to current directory/")
    print("Done.")


if __name__ == "__main__":
    main()
