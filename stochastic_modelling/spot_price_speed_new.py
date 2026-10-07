# -*- coding: utf-8 -*-
# Spot price MRS model — Bakke & Fleten (2016) / Janczura & Weron (2012).

import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from multiprocessing import Pool
from numba import njit
from tqdm import tqdm

CSV_FILE      = os.path.join(os.path.dirname(__file__), "input/DA_NO2_NOK.csv")
DATE_COL      = "Timestamp"
PRICE_COL     = "Price_DA"
# Long-term (GOU) parameters — tune against NVE long-run price forecasts.
GOU_THETA = 570    # long-run mean  (60øre/MW we snittet på NVE langtidsanalyser for 2050 - snitt av alle værscenario))
GOU_KAPPA = 0.05  # mean-reversion speed  (ln(20)/9 ≈ 0.33 → near mean by 2035)
GOU_SIGMA = 0.2  # annual volatility of the long-term level

# Spike/drop sigma decay:  σ(t) = σ_floor + (σ_0 - σ_floor) * exp(-λt)
# Spike and drop sigmas are inflated by crisis data; decay them toward a floor.
# Base regime sigma is left constant (it is already mean-reverting in Z-space).
SPIKE_DECAY_RATE = 0.1   # λ — annual decay rate
SPIKE_FLOOR_FRAC = 0.2  # σ_floor = SPIKE_FLOOR_FRAC × σ_calibrated

# =============================================================================
# SECTION 2 — THREE-REGIME MRS MODEL
# =============================================================================

@njit(cache=True)
def _simulate_core(n_steps, n_paths, x0,
                   P, rho,
                   a1, b1, g1,
                   s1_all, s2_all, s3_all,
                   a2, a3, q2, q3):
    
    # JIT-compiled simulation core (runs at C speed under Numba).
    paths       = np.zeros((n_steps + 1, n_paths))
    base_latent = np.zeros((n_steps + 1, n_paths))   # latent base-regime process
    regimes     = np.zeros((n_steps + 1, n_paths), dtype=np.int64)

    for p in range(n_paths):
        paths[0, p]       = x0
        base_latent[0, p] = x0

    # Draw initial regime from stationary distribution ρ (Eq. 3)
    rho_cum = np.cumsum(rho)
    u_init  = np.random.uniform(0.0, 1.0, n_paths)
    for p in range(n_paths):
        for r in range(3):
            if u_init[p] <= rho_cum[r]:
                regimes[0, p] = r
                break

    for t in range(1, n_steps + 1):
        s1_t = s1_all[t - 1]
        s2_t = s2_all[t - 1]
        s3_t = s3_all[t - 1]

        # Markov chain step — draw next regime from row P[prev_r, :] (Eq. 3)
        u_mc = np.random.uniform(0.0, 1.0, n_paths)
        for p in range(n_paths):
            prev_r = regimes[t - 1, p]
            cum = 0.0
            for r in range(3):
                cum += P[prev_r, r]
                if u_mc[p] <= cum:
                    regimes[t, p] = r
                    break

        # Base regime: CKLS discretisation (Eq. 4)
        #   X_{t,1} = α + (1-β)X_{t-1,1} + σ|X_{t-1,1}|^γ ε_t
        eps1 = np.random.standard_normal(n_paths)
        for p in range(n_paths):
            bl = base_latent[t - 1, p]
            base_latent[t, p] = a1 + (1.0 - b1) * bl + s1_t * (abs(bl) ** g1) * eps1[p]

        # Spike regime: log(X - Q75) ~ N(α2, σ2)  (Eq. 5)
        # Drop  regime: log(Q25 - X) ~ N(α3, σ3)  (Eq. 6)
        eps2 = np.random.standard_normal(n_paths)
        eps3 = np.random.standard_normal(n_paths)
        for p in range(n_paths):
            spike = q2 + np.exp(a2 + s2_t * eps2[p])
            drop  = q3 - np.exp(a3 + s3_t * eps3[p])
            r = regimes[t, p]
            if r == 0:
                paths[t, p] = base_latent[t, p]
            elif r == 1:
                paths[t, p] = spike
            else:
                paths[t, p] = drop

    return paths, base_latent, regimes


def simulate_geo_ou(L0, theta, kappa, sigma_L, n_steps, n_paths=1, dt=1 / 8760):
    # Simulate the Geometric OU long-term factor L_t forward from L0.
    dt_day  = dt * 24
    sqrt_dt = np.sqrt(dt_day)
    n_days  = n_steps // 24 + 1
    eps     = np.random.standard_normal((n_days, n_paths))

    L_daily    = np.empty((n_days + 1, n_paths))
    L_daily[0] = L0
    for t in range(1, n_days + 1):
        dL         = (kappa * (theta - L_daily[t - 1]) * dt_day
                      + sigma_L * L_daily[t - 1] * eps[t - 1] * sqrt_dt)
        L_daily[t] = np.maximum(L_daily[t - 1] + dL, 0.0)

    return np.repeat(L_daily, 24, axis=0)[:n_steps + 1]


class MRSModel:
    def __init__(self, params_base, params_spike, params_drop,
                 transition_matrix, quantiles, initial_probs=None):
        self.pb = params_base
        self.ps = params_spike
        self.pd = params_drop
        self.P  = np.array(transition_matrix)
        self.q2 = quantiles['q2']
        self.q3 = quantiles['q3']

        # COMMENT : gidder ikke hedge mot slike scenarioer. 
        if initial_probs is None:
            # Compute stationary distribution π from P  (solve πP = π)
            vals, vecs = np.linalg.eig(self.P.T)
            stat = np.abs(vecs[:, np.argmin(np.abs(vals - 1))])
            self.rho = stat / stat.sum()
        else:
            self.rho = np.array(initial_probs)

    #Unpacking stored parameters from self 
    def simulate(self, x0, n_steps, n_paths=1):
        """Simulate n_paths forward paths of Z_t starting from x0."""
        a1, b1 = self.pb['alpha'], self.pb['beta']
        s1, g1 = self.pb['sigma'], self.pb['gamma']
        a2, s2 = self.ps['alpha'], self.ps['sigma']
        a3, s3 = self.pd['alpha'], self.pd['sigma']

        # Hvordan de ulike sigmaene for base, drop, spike avtar som følge av spike decay og spike frac. 
        s1_all = np.full(n_steps, s1)  # constant
        t_years = np.arange(1, n_steps + 1) / 8760
        #s1_inf = SPIKE_FLOOR_FRAC * s1
        #s1_all = s1_inf + (s1 - s1_inf) * np.exp(-SPIKE_DECAY_RATE * t_years)
        s2_inf  = SPIKE_FLOOR_FRAC * s2
        s3_inf  = SPIKE_FLOOR_FRAC * s3
        s2_all  = s2_inf + (s2 - s2_inf) * np.exp(-SPIKE_DECAY_RATE * t_years)
        s3_all  = s3_inf + (s3 - s3_inf) * np.exp(-SPIKE_DECAY_RATE * t_years)

        return _simulate_core(
            n_steps, n_paths, float(x0),
            self.P, self.rho,
            a1, b1, g1,
            s1_all, s2_all, s3_all,
            a2, a3, self.q2, self.q3,
        )


# =============================================================================
# SECTION 3 — EM CALIBRATION  (Janczura & Weron 2012)
# =============================================================================
class Calibrator:
    """
    The EM algorithm alternates between:
      E-step — given parameters, compute smoothed regime probabilities
               using the forward-backward (Kalman-style) filter
      M-step — given regime probabilities, update parameters analytically
               (base: Eqs. 7-9;  spike/drop: weighted log-normal MLE;
                transitions: Eq. 10)
    """

    def __init__(self, max_iter=200, tol=1e-2, verbose=True):
        self.max_iter = max_iter
        self.tol      = tol
        self.verbose  = verbose

    # ----------------------- PDF helpers -------------------------

    def _pdf_base(self, xt, x_tilde_prev, alpha1, beta1, sigma1, gamma1):
        # Base regime PDF: N(α + (1-β)x̃_{t-1}, σ|x̃_{t-1}|^γ)  (Eq. 11)
        if not np.isfinite(x_tilde_prev):
            return 1e-300
        mean = alpha1 + (1 - beta1) * x_tilde_prev
        std  = sigma1 * (np.abs(x_tilde_prev) ** gamma1) + 1e-8
        if not np.isfinite(std) or std < 1e-8:
            std = 1e-8
        return norm.pdf(xt, loc=mean, scale=std)

    def _pdf_spike_vec(self, X, alpha2, sigma2, q2):
        # Shifted log-normal PDF for all observations (Eq. 5)
        result = np.zeros(len(X))
        mask   = X > q2
        if mask.any():
            shifted      = np.maximum(X[mask] - q2, 1e-10)
            result[mask] = norm.pdf(np.log(shifted), loc=alpha2,
                                    scale=sigma2 + 1e-8) / shifted
        return result

    def _pdf_drop_vec(self, X, alpha3, sigma3, q3):
        # Shifted inverse log-normal PDF for all observations (Eq. 6).
        result = np.zeros(len(X))
        mask   = X < q3
        if mask.any():
            shifted      = np.maximum(q3 - X[mask], 1e-10)
            result[mask] = norm.pdf(np.log(shifted), loc=alpha3,
                                    scale=sigma3 + 1e-8) / shifted
        return result

    # ----------------------- E-STEP: forward filter + backward smoother -----------------------

    def _e_step(self, X, theta):
        T      = len(X)
        alpha1 = theta['alpha1']; beta1  = theta['beta1']
        sigma1 = theta['sigma1']; gamma1 = theta['gamma1']
        alpha2 = theta['alpha2']; sigma2 = theta['sigma2']
        alpha3 = theta['alpha3']; sigma3 = theta['sigma3']
        P      = theta['P'];      rho    = theta['rho']
        q2     = theta['q2'];     q3     = theta['q3']

        filt      = np.zeros((T, 3))
        filt_pred = np.zeros((T, 3))
        x_tilde   = np.zeros(T)
        loglik    = 0.0

        filt_pred[0] = rho
        x_tilde[0]   = X[0]

        # Pre-compute spike/drop PDFs (vectorised — no x̃ dependency)
        f_spike = self._pdf_spike_vec(X, alpha2, sigma2, q2)
        f_drop  = self._pdf_drop_vec(X, alpha3, sigma3, q3)
        x_range = np.abs(X).max() * 10

        # Forward filter (sequential — x̃ recursion requires previous step)
        for t in range(T):
            x_prev_tilde = x_tilde[t - 1] if t > 0 else X[0]
            f1  = self._pdf_base(X[t], x_prev_tilde, alpha1, beta1, sigma1, gamma1)
            f   = np.where(np.isfinite([f1, f_spike[t], f_drop[t]]),
                           [f1, f_spike[t], f_drop[t]], 1e-300)
            raw   = filt_pred[t] * f
            denom = raw.sum()
            filt[t]  = raw / denom if denom >= 1e-300 else filt_pred[t]
            loglik  += np.log(max(denom, 1e-300))
            if t < T - 1:
                filt_pred[t + 1] = np.dot(P.T, filt[t])

            # x̃ recursion (Eq. 12): weight observed X_t by P(base) + propagate
            if t == 0:
                x_tilde[0] = X[0]
            else:
                p_in       = filt[t, 0]
                candidate  = (p_in * X[t] +
                              (1.0 - p_in) * (alpha1 + (1 - beta1) * x_tilde[t - 1]))
                x_tilde[t] = float(np.clip(candidate, -x_range, x_range))

        # Backward smoother — fully vectorised (Kim 1994)
        smooth      = np.zeros((T, 3))
        smooth[T-1] = filt[T-1]
        for t in range(T - 2, -1, -1):
            ratio     = smooth[t + 1] / np.maximum(filt_pred[t + 1], 1e-300)
            smooth[t] = filt[t] * np.dot(P, ratio)
            s         = smooth[t].sum()
            smooth[t] = smooth[t] / s if s > 1e-300 else filt[t]

        return filt, filt_pred, smooth, x_tilde, loglik

    
    
    # ----------------------- M-STEP -----------------------

    def _m_step_base_fixed_gamma(self, X, smooth, x_tilde, gamma1, theta_prev):
        """Analytical α1, β1, σ1 for a fixed γ1 (Eqs. 7, 8, 9)."""
        w     = smooth[1:, 0]
        if self._obs_weights is not None:
            w = w * self._obs_weights[1:]
        xt    = X[1:]
        xt_1  = x_tilde[:-1]
        abs_x = np.abs(xt_1) + 1e-8
        v     = w * abs_x ** (-2 * gamma1)
        S_v   = v.sum()
        W     = w.sum()

        if S_v < 1e-10 or not np.isfinite(S_v):
            if self.verbose:
                print("  [guard] base regime collapsed - keeping previous params")
            return theta_prev['alpha1'], theta_prev['beta1'], theta_prev['sigma1']

        # β̂ (Eq. 8)
        delta    = xt - xt_1
        d_bar    = (v * delta).sum() / S_v
        x_bar    = (v * xt_1).sum()  / S_v
        u        = w * xt_1 * abs_x ** (-2 * gamma1)
        num_beta = (u * (delta - d_bar)).sum()
        den_beta = (u * (xt_1 - x_bar)).sum()
        beta1    = (num_beta / den_beta if np.abs(den_beta) >= 1e-12
                    else theta_prev['beta1'])
        beta1    = float(np.clip(beta1, 0.05, 0.9999))

        # α̂ (Eq. 7)
        alpha1 = float((v * (xt - (1 - beta1) * xt_1)).sum() / S_v)
        if not np.isfinite(alpha1):
            alpha1 = theta_prev['alpha1']

        # σ̂ (Eq. 9)
        resid     = xt - alpha1 - (1 - beta1) * xt_1
        sigma1_sq = float((v * resid ** 2).sum() / W)
        sigma1    = (np.sqrt(sigma1_sq) if np.isfinite(sigma1_sq) and sigma1_sq >= 1e-8
                     else theta_prev['sigma1'])

        return alpha1, beta1, sigma1

    def _m_step_base(self, X, smooth, x_tilde, gamma1_init, theta_prev):
        """Full base M-step: 1-D search over γ1, analytical Eqs. 7-9 per candidate."""
        xt    = X[1:]
        xt_1  = x_tilde[:-1]
        w     = smooth[1:, 0]
        if self._obs_weights is not None:
            w = w * self._obs_weights[1:]
        abs_x = np.abs(xt_1) + 1e-8

        def neg_loglik_gamma(gamma):
            a, b, s = self._m_step_base_fixed_gamma(X, smooth, x_tilde, gamma, theta_prev)
            std_t = s * abs_x ** gamma + 1e-8
            ll    = w * norm.logpdf(xt, loc=a + (1 - b) * xt_1, scale=std_t)
            return -(np.where(np.isfinite(ll), ll, -1e10)).sum()

        result = minimize_scalar(neg_loglik_gamma, bounds=(0.0, 2.0), method='bounded')
        gamma1 = float(result.x)
        alpha1, beta1, sigma1 = self._m_step_base_fixed_gamma(
            X, smooth, x_tilde, gamma1, theta_prev)
        return alpha1, beta1, sigma1, gamma1

    def _m_step_spike(self, X, smooth, q2, theta_prev):
        """Spike M-step: weighted MLE for shifted log-normal (Eq. 5)."""
        w2 = smooth[:, 1]
        if self._obs_weights is not None:
            w2 = w2 * self._obs_weights
        mask = X > q2
        W2   = w2[mask].sum()
        if mask.sum() < 2 or W2 < 1e-10:
            if self.verbose:
                print("  [guard] spike regime collapsed - keeping previous params")
            return theta_prev['alpha2'], theta_prev['sigma2']

        log_shifted = np.log(np.maximum(X[mask] - q2, 1e-10))
        w           = w2[mask]
        alpha2      = float((w * log_shifted).sum() / W2)
        var2        = float((w * (log_shifted - alpha2) ** 2).sum() / W2)
        sigma2      = np.sqrt(max(var2, 1e-8)) if np.isfinite(var2) else theta_prev['sigma2']
        return alpha2, sigma2

    def _m_step_drop(self, X, smooth, q3, theta_prev):
        """Drop M-step: weighted MLE for shifted inverse log-normal (Eq. 6)."""
        w3 = smooth[:, 2]
        if self._obs_weights is not None:
            w3 = w3 * self._obs_weights
        mask = X < q3
        W3   = w3[mask].sum()
        if mask.sum() < 2 or W3 < 1e-10:
            if self.verbose:
                print("  [guard] drop regime collapsed - keeping previous params")
            return theta_prev['alpha3'], theta_prev['sigma3']

        log_shifted = np.log(np.maximum(q3 - X[mask], 1e-10))
        w           = w3[mask]
        alpha3      = float((w * log_shifted).sum() / W3)
        var3        = float((w * (log_shifted - alpha3) ** 2).sum() / W3)
        sigma3      = np.sqrt(max(var3, 1e-8)) if np.isfinite(var3) else theta_prev['sigma3']
        return alpha3, sigma3

    def _m_step_transitions(self, filt, filt_pred, smooth, P_old):
        """Transition matrix M-step — vectorised (Eq. 10)."""
        ratio = smooth[1:] / np.maximum(filt_pred[1:], 1e-300)  # (T-1, 3)
        f_weighted = filt[:-1]
        if self._obs_weights is not None:
            f_weighted = f_weighted * self._obs_weights[:-1, np.newaxis]
        joint = P_old * np.dot(f_weighted.T, ratio)              # (3, 3)
        row_sums = joint.sum(axis=1, keepdims=True)
        return np.where(row_sums > 1e-300, joint / row_sums, P_old)

    # ------------------------------------------------------------------
    # MAIN FIT METHOD
    # ------------------------------------------------------------------

    def fit(self, X, theta_init, obs_weights=None,
            spike_pct=90, drop_pct=10):
        """
        obs_weights : array-like of length len(X), or None.
            Positive weights applied in every M-step.  Values <1 downweight
            observations (e.g. assign 0.3 to crisis-period data so the
            calibrated parameters lean toward the pre-crisis regime).
            Weights are normalised to sum to len(X) so the effective sample
            size is preserved.
        spike_pct   : upper percentile for spike threshold (default 90).
            Only observations above this quantile can be assigned to the spike
            regime. Lower values make spikes less rare; higher makes them rarer.
        drop_pct    : lower percentile for drop threshold (default 10).
            Only observations below this quantile can be assigned to the drop
            regime.
        """
        if obs_weights is not None:
            ow = np.asarray(obs_weights, dtype=float)
            ow = ow * (len(ow) / ow.sum())   # normalise: mean weight = 1
            self._obs_weights = ow
        else:
            self._obs_weights = None

        theta        = dict(theta_init)
        theta['P']   = np.array(theta['P'],   dtype=float)
        theta['rho'] = np.array(theta['rho'], dtype=float)
        theta['q2']  = float(np.percentile(X, spike_pct))
        theta['q3']  = float(np.percentile(X, drop_pct))
        best_theta  = dict(theta)
        best_loglik = -np.inf
        logliks     = []

        for iteration in tqdm(range(self.max_iter), desc="EM calibration"):
            filt, filt_pred, smooth, x_tilde, loglik = self._e_step(X, theta)

            # Regime-collapse diagnostic — printed every iteration
            w_base  = smooth[:, 0].mean()
            w_spike = smooth[:, 1].mean()
            w_drop  = smooth[:, 2].mean()
            collapsed = [r for r, w in [("base", w_base), ("spike", w_spike), ("drop", w_drop)]
                         if w < 0.005]
            collapse_flag = "  *** COLLAPSED: %s ***" % ", ".join(collapsed) if collapsed else ""
            if self.verbose:
                print("  [iter %3d] loglik=%.2f  w_base=%.3f  w_spike=%.3f  w_drop=%.3f%s" % (
                    iteration, loglik, w_base, w_spike, w_drop, collapse_flag))

            if not np.isfinite(loglik):
                return best_theta, logliks
            logliks.append(loglik)
            if loglik > best_loglik:
                best_loglik = loglik
                best_theta  = dict(theta)
            if iteration > 0 and 0 <= (loglik - logliks[-2]) < self.tol:
                break
            theta_prev = dict(theta)
            alpha1, beta1, sigma1, gamma1 = self._m_step_base(X, smooth, x_tilde, theta['gamma1'], theta_prev)
            alpha2, sigma2                = self._m_step_spike(X, smooth, theta['q2'], theta_prev)
            alpha3, sigma3                = self._m_step_drop(X,  smooth, theta['q3'], theta_prev)
            P_new                         = self._m_step_transitions(filt, filt_pred, smooth, theta['P'])
            theta.update({
                'alpha1': alpha1, 'beta1': beta1,   'sigma1': sigma1, 'gamma1': gamma1,
                'alpha2': alpha2, 'sigma2': sigma2,
                'alpha3': alpha3, 'sigma3': sigma3,
                'P': P_new, 'rho': smooth[0].copy(),
            })
        return theta, logliks

def build_model(theta):
    """Build a MRSModel from a calibrated theta dict (output of Calibrator.fit())."""
    return MRSModel(
        params_base  = {'alpha': theta['alpha1'], 'beta':  theta['beta1'],
                        'sigma': theta['sigma1'], 'gamma': theta['gamma1']},
        params_spike = {'alpha': theta['alpha2'], 'sigma': theta['sigma2']},
        params_drop  = {'alpha': theta['alpha3'], 'sigma': theta['sigma3']},
        transition_matrix = theta['P'].tolist(),
        quantiles    = {'q2': theta['q2'], 'q3': theta['q3']},
        initial_probs = theta['rho'].tolist(),
    )

# =============================================================================
# SECTION 4 — DESEASONALISATION  (Bakke & Fleten 2016, Eq. 1: ft = st + dt + wt)
# =============================================================================

def fit_sinusoidal_ltsc(t, prices, n_harmonics=2, period=8760):
    # Fit the long-term seasonal component as OLS sinusoids (Janczura & Weron 2010)
    cols = [np.ones(len(t))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t / period))
        cols.append(np.sin(2 * np.pi * k * t / period))
    A      = np.column_stack(cols)
    coeffs = np.linalg.lstsq(A, prices, rcond=None)[0]
    return coeffs, np.dot(A, coeffs)


def extrapolate_ltsc(t_future, coeffs, n_harmonics=None, period=8760):
    # Evaluate the fitted LTSC at future time indices (for re-seasonalisation).
    # n_harmonics is inferred from coeffs length if not provided: len = 1 + 2*n_harmonics
    if n_harmonics is None:
        n_harmonics = (len(coeffs) - 1) // 2
    cols = [np.ones(len(t_future))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t_future / period))
        cols.append(np.sin(2 * np.pi * k * t_future / period))
    return np.dot(np.column_stack(cols), coeffs)


def deseasonalise(df, n_harmonics=2, hours_per_year=8760):
    prices = df['price'].values
    t = np.arange(len(prices), dtype=float)
    ltsc_coeffs, ltsc_vals = fit_sinusoidal_ltsc(t, prices, n_harmonics, hours_per_year)
    df['ltsc']    = ltsc_vals
    price_no_ltsc = prices - ltsc_vals

    if hasattr(df.index, 'hour'):
        hour_med    = pd.Series(price_no_ltsc, index=df.index).groupby(df.index.hour).median()
        df['daily'] = df.index.hour.map(hour_med)
    else:
        df['daily'] = 0.0

    price_no_daily = price_no_ltsc - df['daily'].values

    if hasattr(df.index, 'dayofweek'):
        dow_med      = pd.Series(price_no_daily, index=df.index).groupby(df.index.dayofweek).median()
        df['weekly'] = df.index.dayofweek.map(dow_med)
    else:
        df['weekly'] = 0.0

    Xt_raw  = price_no_daily - df['weekly'].values
    shift   = prices.mean() - Xt_raw.mean()
    df['X'] = Xt_raw + shift
    return df, ltsc_coeffs

# =============================================================================
# SECTION 5 — DATA LOADING
# =============================================================================

def load_prices(csv_file, date_col=DATE_COL, price_col=PRICE_COL):
    """Load hourly electricity spot prices from CSV."""
    df = pd.read_csv(csv_file)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        df.index = df[date_col]
    df = df.rename(columns={price_col: 'price'})
    return df

# =============================================================================
# SECTION 6 — FORECASTING  (parallel Monte Carlo)
# =============================================================================

# for å kjøre flere simuleringer i parallell/ stykkes opp i batches. 
def _simulate_worker(args):
    model, z0, n_steps, n_paths_batch, seed = args
    np.random.seed(seed)
    paths, _, _ = model.simulate(z0, n_steps, n_paths_batch)
    return paths


def forecast_parallel(model, z0, L0, n_forecast=131400,
                      n_simulations=1, n_workers=None):
    # Splitter arbeidet på ulike workers
    if n_workers is None:
        n_workers = min(n_simulations, os.cpu_count())
    base, remainder = divmod(n_simulations, n_workers)
    batch_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    seeds = np.random.randint(0, 2 ** 31, size=n_workers).tolist() # TODO vurdere om vi faktisk skal ha helt randome seeds hver gang
    #splitter opp simuleringene i batches, og for hver batch lages et sett med argumenter som skal sendes til workerne.
    worker_args = [(model, z0, n_forecast, bs, seed) for bs, seed in zip(batch_sizes, seeds) if bs > 0]
    # Setter sammen arbeidet til de ulike workerne. 
    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(_simulate_worker, worker_args), 
                            total=len(worker_args),
                            desc="Simulating paths"))
    Z_paths = np.concatenate(results, axis=1)
    # Trenger ikke kjøre long term faktoren i parallell. 
    L_paths = simulate_geo_ou(L0, GOU_THETA, GOU_KAPPA, GOU_SIGMA, n_forecast, n_simulations)
    return L_paths + Z_paths


# =============================================================================
# SECTION 7 — OUTPUT / PERSISTENCE
# =============================================================================

def _reseasonalise(paths, df, ltsc_coeffs, start_dt=None, freq='h'):
    # Reverse deseasonalise(): add back LTSC + daily + weekly components.
    n_steps = paths.shape[0] - 1
    if start_dt is None:
        start_dt = df.index[-1] + pd.tseries.frequencies.to_offset(freq)
    future_dates = pd.date_range(start_dt, periods=n_steps, freq=freq)
    t_offset     = len(df)
    t_future     = np.arange(t_offset, t_offset + n_steps, dtype=float)
    ltsc_future  = extrapolate_ltsc(t_future, ltsc_coeffs)
    hour_med = df.groupby(df.index.hour)['daily'].mean()
    dow_med  = df.groupby(df.index.dayofweek)['weekly'].mean()
    daily_future  = future_dates.hour.map(hour_med).values.astype(float)
    weekly_future = future_dates.dayofweek.map(dow_med).values.astype(float)
    shift    = float((df['X'] + df['ltsc'] + df['daily'] + df['weekly'] - df['price']).mean())
    seasonal = ltsc_future + daily_future + weekly_future - shift  # (n_steps,)
    reseason = paths[1:] + seasonal[:, np.newaxis]                 # (n_steps, n_paths)
    return reseason, future_dates

def save_paths_to_csv(paths, df, ltsc_coeffs, out_path,
                      start_dt=None, freq='h', inflation_rate=0.0,
                      price_floor=None, price_cap=None):
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    reseason, future_dates = _reseasonalise(paths, df, ltsc_coeffs, start_dt, freq)
    if inflation_rate != 0.0:
        t_years = np.arange(1, n_steps + 1) / 8760
        infl    = (1.0 + inflation_rate) ** t_years
        reseason = reseason * infl[:, np.newaxis]
    if price_floor is not None or price_cap is not None:
        reseason = np.clip(reseason, price_floor, price_cap)
    cols   = ['path_%d' % i for i in range(n_paths)]
    out_df = pd.DataFrame(reseason, index=future_dates, columns=cols)
    out_df.index.name = 'timestamp'
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if out_path.endswith('.parquet'):
        out_df.to_parquet(out_path)
    else:
        out_df.to_csv(out_path)
    return out_df

def save_params_to_csv(theta, out_path, label=None):
    P   = np.array(theta['P'])

    # Solve π P = π, sum(π) = 1  (avoids np.linalg.eig complex-eigenvector pitfalls)
    A   = np.vstack([P.T - np.eye(3), np.ones((1, 3))])
    b   = np.zeros(4); b[-1] = 1.0
    rho_raw, *_ = np.linalg.lstsq(A, b, rcond=None)
    rho = np.abs(rho_raw) / np.abs(rho_raw).sum()

    row = {}
    if label is not None:
        row['series'] = label
    row.update({
        'alpha1':        round(theta['alpha1'], 6),
        'beta1':         round(theta['beta1'],  6),
        'sigma1':        round(theta['sigma1'], 6),
        'gamma1':        round(theta['gamma1'], 6),
        'long_run_mean': round(theta['alpha1'] / theta['beta1'], 4),
        'alpha2':        round(theta['alpha2'], 6),
        'sigma2':        round(theta['sigma2'], 6),
        'q2':            round(theta['q2'],     4),
        'alpha3':        round(theta['alpha3'], 6),
        'sigma3':        round(theta['sigma3'], 6),
        'q3':            round(theta['q3'],     4),
    })
    for i in range(3):
        for j in range(3):
            row['P_%d%d' % (i, j)] = round(float(P[i, j]), 6)
    for i in range(3):
        row['rho_%d' % i] = round(float(rho[i]), 6)

    row_df = pd.DataFrame([row])
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    row_df.to_csv(out_path, index=False)
    return row_df

# TODO har satt opp initielle parametere - men kan spørre SE om å bruke Kmeans? eller finne ut hva andre gjør. 
INITIAL_PARAMS = {
    'alpha1': 2.0,   # α1 ≈ β1 × long-run mean of Z  (should be near 0 after L removal)
    'beta1':  0.40,  # mean-reversion speed
    'sigma1': 3.0,   # base regime volatility
    'gamma1': 0.0,   # elasticity of variance (0 → Vasicek; 0.5 → CIR)

    'alpha2': 3.3,   # log-scale spike location
    'sigma2': 1.0,   # log-scale spike spread

    'alpha3': 3.0,   # log-scale drop location
    'sigma3': 0.25,  # log-scale drop spread

    'P': [[0.97, 0.015, 0.015],
          [0.17, 0.83,  0.00 ],
          [0.09, 0.00,  0.91 ]],

    'rho': [0.92, 0.04, 0.04],
}

# =============================================================================
# MAIN
# =============================================================================