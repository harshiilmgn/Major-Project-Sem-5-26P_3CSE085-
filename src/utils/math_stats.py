"""
Mathematical and Statistical Utility Functions
Implements non-parametric tests, convergence metrics, and econometric estimators.
"""

import numpy as np
from scipy import stats


def calculate_cagr(start_val, end_val, periods):
    """Calculates Compound Annual Growth Rate."""
    if start_val is None or end_val is None or start_val <= 0 or end_val <= 0 or periods <= 0:
        return np.nan
    return (end_val / start_val) ** (1.0 / periods) - 1.0


def log_linear_trend(values, years=None):
    """
    Fits ln(Y_t) = alpha + beta * t + e_t.
    Returns: beta, growth_rate (exp(beta) - 1), r_squared, p_value, stderr
    """
    valid_idx = [i for i, v in enumerate(values) if v is not None and not np.isnan(v) and v > 0]
    if len(valid_idx) < 3:
        return {"beta": np.nan, "growth_rate": np.nan, "r2": np.nan, "p_value": np.nan, "stderr": np.nan}
    
    y = np.log([values[i] for i in valid_idx])
    x = np.arange(len(valid_idx)) if years is None else np.array([years[i] for i in valid_idx], dtype=float)
    x = x - x.min()
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    growth_rate = np.exp(slope) - 1.0
    return {
        "beta": float(slope),
        "growth_rate": float(growth_rate),
        "r2": float(r_value ** 2),
        "p_value": float(p_value),
        "stderr": float(std_err)
    }


def mann_kendall_test(x):
    """
    Non-parametric Mann-Kendall monotonic trend test.
    Returns: tau, score (S), var_s, z, p_value, trend ('increasing', 'decreasing', 'no trend')
    """
    x = np.array([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n < 4:
        return {"trend": "insufficient_data", "p_value": np.nan, "tau": np.nan, "z": np.nan, "s": np.nan}
    
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
            
    # Calculate tie adjustments
    unique_vals, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
        
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    
    if p_value < 0.05:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no trend"
        
    return {
        "trend": trend,
        "p_value": float(p_value),
        "tau": float(tau),
        "z": float(z),
        "s": float(s)
    }


def sens_slope(x):
    """
    Computes Sen's slope estimator (median of all pairwise slopes).
    """
    x = np.array([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    return float(np.median(slopes))


def sigma_convergence(state_series_by_year):
    """
    Measures dispersion (Coefficient of Variation) over time.
    state_series_by_year: dict {year: [val_state1, val_state2, ...]}
    Returns: dict {year: {'mean': ..., 'std': ..., 'cv': ...}}
    """
    results = {}
    for year, vals in state_series_by_year.items():
        clean_vals = [v for v in vals if v is not None and not np.isnan(v) and v > 0]
        if len(clean_vals) > 0:
            mean = np.mean(clean_vals)
            std = np.std(clean_vals, ddof=1)
            cv = (std / mean) if mean != 0 else np.nan
            results[year] = {"mean": float(mean), "std": float(std), "cv": float(cv)}
        else:
            results[year] = {"mean": np.nan, "std": np.nan, "cv": np.nan}
    return results


def beta_convergence(initial_vals, final_vals, periods):
    """
    Unconditional Beta convergence:
    (1/T) * ln(y_iT / y_i0) = alpha + beta * ln(y_i0) + eps_i
    Negative beta implies convergence (poorer states grow faster).
    Speed of convergence lambda = -ln(1 + beta * T) / T.
    """
    valid = []
    for y0, yT in zip(initial_vals, final_vals):
        if y0 is not None and yT is not None and y0 > 0 and yT > 0:
            valid.append((y0, yT))
    if len(valid) < 5:
        return {"beta": np.nan, "speed_lambda": np.nan, "p_value": np.nan, "converging": False}
    
    y0_arr = np.array([v[0] for v in valid], dtype=float)
    yT_arr = np.array([v[1] for v in valid], dtype=float)
    
    growth_annualized = (np.log(yT_arr) - np.log(y0_arr)) / float(periods)
    ln_y0 = np.log(y0_arr)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_y0, growth_annualized)
    
    if (1.0 + slope * periods) > 0:
        lambda_speed = -np.log(1.0 + slope * periods) / periods
    else:
        lambda_speed = np.nan
        
    return {
        "beta": float(slope),
        "r2": float(r_value ** 2),
        "p_value": float(p_value),
        "stderr": float(std_err),
        "speed_lambda": float(lambda_speed) if not np.isnan(lambda_speed) else None,
        "converging": bool(slope < 0 and p_value < 0.05)
    }


def robust_z_scores(values):
    """
    Computes Robust Z-scores using Median and Median Absolute Deviation (MAD).
    Z_rob = 0.6745 * (x - median) / MAD
    Thresholds: |Z_rob| > 3.0 indicates significant anomaly.
    """
    arr = np.array(values, dtype=float)
    median = np.nanmedian(arr)
    diff = np.abs(arr - median)
    mad = np.nanmedian(diff)
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(arr)
    z_rob = 0.6745 * (arr - median) / mad
    return z_rob


def compute_dissimilarity_index(shares_t1, shares_t2):
    """
    Dissimilarity Index between two budget distributions:
    D = 0.5 * sum(|s_{i, t1} - s_{i, t2}|)
    """
    s1 = np.array(shares_t1, dtype=float)
    s2 = np.array(shares_t2, dtype=float)
    if np.sum(s1) > 0:
        s1 = s1 / np.sum(s1)
    if np.sum(s2) > 0:
        s2 = s2 / np.sum(s2)
    return float(0.5 * np.sum(np.abs(s1 - s2)))
