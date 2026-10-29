"""
Econometric Panel Model Estimator
Implements Fixed Effects (Within Estimator), Random Effects (GLS/Between),
Pooled OLS, and the Hausman Specification Test.
"""

import csv
from collections import defaultdict
import numpy as np
from scipy import stats


class PanelRegressionEngine:
    def __init__(self, panel_csv_path: str):
        self.records = []
        with open(panel_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.records.append(r)
                
        # Group by state
        self.by_state = defaultdict(list)
        for r in self.records:
            self.by_state[r["state_code"]].append(r)

    def estimate_model(self, dep_var: str = "ln_sse_pc", indep_vars: list = None) -> dict:
        """
        Estimates Fixed Effects, Random Effects, and Pooled OLS models.
        Runs the Hausman test to guide model selection.
        """
        if indep_vars is None:
            indep_vars = ["ln_gsdp_pc", "otr_gsdp", "dev_rr", "fd_gsdp", "covid_shock"]
            
        # Prepare data matrices
        # Extract variables
        state_ids = []
        y_list = []
        X_list = []
        
        for r in self.records:
            state_ids.append(r["state_code"])
            
            # Dependent var
            if dep_var == "ln_sse_pc":
                y_val = np.log(float(r["sse_pc"]))
            elif dep_var == "sse_te":
                y_val = float(r["sse_te"])
            elif dep_var == "sse_gsdp":
                y_val = float(r["sse_gsdp"])
            else:
                y_val = float(r[dep_var])
            y_list.append(y_val)
            
            # Independent vars
            row_x = []
            for var in indep_vars:
                if var == "ln_gsdp_pc":
                    row_x.append(np.log(float(r["gsdp_pc_real"])))
                else:
                    row_x.append(float(r[var]))
            X_list.append(row_x)
            
        y = np.array(y_list, dtype=float)
        X = np.array(X_list, dtype=float)
        states = np.array(state_ids)
        unique_states = np.unique(states)
        
        N = len(unique_states)
        NT = len(y)
        K = X.shape[1]
        
        # 1. Pooled OLS
        X_pooled = np.column_stack([np.ones(NT), X])
        beta_pooled = np.linalg.pinv(X_pooled.T @ X_pooled) @ (X_pooled.T @ y)
        resid_pooled = y - X_pooled @ beta_pooled
        s2_pooled = np.sum(resid_pooled ** 2) / (NT - K - 1)
        cov_pooled = s2_pooled * np.linalg.pinv(X_pooled.T @ X_pooled)
        se_pooled = np.sqrt(np.diag(cov_pooled))
        
        # 2. Fixed Effects (Within Transformation)
        y_within = np.zeros_like(y)
        X_within = np.zeros_like(X)
        
        for s in unique_states:
            idx = (states == s)
            y_within[idx] = y[idx] - np.mean(y[idx])
            X_within[idx] = X[idx] - np.mean(X[idx], axis=0)
            
        beta_fe = np.linalg.pinv(X_within.T @ X_within) @ (X_within.T @ y_within)
        resid_fe = y_within - X_within @ beta_fe
        deg_free_fe = NT - N - K
        s2_fe = np.sum(resid_fe ** 2) / deg_free_fe
        cov_fe = s2_fe * np.linalg.pinv(X_within.T @ X_within)
        se_fe = np.sqrt(np.maximum(0, np.diag(cov_fe)))
        
        # Within R2
        ss_tot_fe = np.sum(y_within ** 2)
        ss_res_fe = np.sum(resid_fe ** 2)
        r2_within = 1.0 - (ss_res_fe / ss_tot_fe) if ss_tot_fe > 0 else 0.0
        
        # 3. Random Effects (GLS Transformation)
        # Calculate theta for quasi-demeaning
        # Group means (Between regression)
        y_between = np.array([np.mean(y[states == s]) for s in unique_states])
        X_between = np.array([np.mean(X[states == s], axis=0) for s in unique_states])
        X_between_const = np.column_stack([np.ones(N), X_between])
        beta_between = np.linalg.pinv(X_between_const.T @ X_between_const) @ (X_between_const.T @ y_between)
        resid_between = y_between - X_between_const @ beta_between
        s2_between = np.sum(resid_between ** 2) / (N - K - 1)
        
        # Variance of idiosyncratic error (s2_e) and individual effect (s2_u)
        s2_e = s2_fe
        T_bar = NT / N
        s2_u = max(0.0, s2_between - (s2_e / T_bar))
        theta = 1.0 - np.sqrt(s2_e / (s2_e + T_bar * s2_u)) if (s2_e + T_bar * s2_u) > 0 else 0.0
        
        # Quasi-demeaning
        y_re = np.zeros_like(y)
        X_re = np.zeros_like(X)
        for s in unique_states:
            idx = (states == s)
            y_re[idx] = y[idx] - theta * np.mean(y[idx])
            X_re[idx] = X[idx] - theta * np.mean(X[idx], axis=0)
            
        const_re = (1.0 - theta) * np.ones(NT)
        X_re_full = np.column_stack([const_re, X_re])
        beta_re_full = np.linalg.pinv(X_re_full.T @ X_re_full) @ (X_re_full.T @ y_re)
        beta_re = beta_re_full[1:] # drop constant for comparison with FE
        
        resid_re = y_re - X_re_full @ beta_re_full
        s2_re = np.sum(resid_re ** 2) / (NT - K - 1)
        cov_re_full = s2_re * np.linalg.pinv(X_re_full.T @ X_re_full)
        cov_re = cov_re_full[1:, 1:]
        se_re = np.sqrt(np.maximum(0, np.diag(cov_re)))
        
        # 4. Hausman Test: H = (b_FE - b_RE)' * (V_FE - V_RE)^(-1) * (b_FE - b_RE)
        diff_beta = beta_fe - beta_re
        diff_cov = cov_fe - cov_re
        try:
            h_stat = float(diff_beta.T @ np.linalg.pinv(diff_cov) @ diff_beta)
            h_stat = max(0.0, h_stat)
            h_p_value = float(1.0 - stats.chi2.cdf(h_stat, df=K))
        except Exception:
            h_stat = np.nan
            h_p_value = np.nan
            
        preferred_model = "Fixed Effects (FE)" if (not np.isnan(h_p_value) and h_p_value < 0.05) else "Random Effects (RE)"
        
        # Format output tables
        fe_summary = []
        for j, var in enumerate(indep_vars):
            coef = beta_fe[j]
            se = se_fe[j]
            t_stat = coef / se if se > 0 else 0
            p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=deg_free_fe))
            fe_summary.append({
                "variable": var,
                "coef": round(float(coef), 4),
                "std_err": round(float(se), 4),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_val), 5),
                "significance": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            })
            
        re_summary = []
        for j, var in enumerate(indep_vars):
            coef = beta_re[j]
            se = se_re[j]
            z_stat = coef / se if se > 0 else 0
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
            re_summary.append({
                "variable": var,
                "coef": round(float(coef), 4),
                "std_err": round(float(se), 4),
                "z_stat": round(float(z_stat), 3),
                "p_value": round(float(p_val), 5),
                "significance": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            })
            
        return {
            "dep_var": dep_var,
            "sample_size": NT,
            "num_states": N,
            "num_regressors": K,
            "within_r2": round(float(r2_within), 4),
            "fixed_effects_results": fe_summary,
            "random_effects_results": re_summary,
            "hausman_test": {
                "chi2_stat": round(float(h_stat), 3) if not np.isnan(h_stat) else None,
                "df": K,
                "p_value": round(float(h_p_value), 5) if not np.isnan(h_p_value) else None,
                "preferred_model": preferred_model,
                "interpretation": "FE is preferred because individual state effects are correlated with regressors (p < 0.05)" if (not np.isnan(h_p_value) and h_p_value < 0.05) else "RE cannot be rejected"
            }
        }
