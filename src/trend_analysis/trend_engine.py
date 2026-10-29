"""
Trend and Convergence Analysis Engine
Implements CAGR across sub-periods, log-linear regressions, Mann-Kendall tests,
Sen's slope, sigma convergence, and beta convergence.
"""

import csv
import json
from collections import defaultdict
import numpy as np
from src.utils.math_stats import (
    calculate_cagr,
    log_linear_trend,
    mann_kendall_test,
    sens_slope,
    sigma_convergence,
    beta_convergence,
    compute_dissimilarity_index
)


class TrendAnalysisEngine:
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
            
        # Sort each state by year_t
        for s in self.by_state:
            self.by_state[s].sort(key=lambda x: int(x["year_t"]))
            
        self.years = sorted(list(set(r["fiscal_year"] for r in self.records)), key=lambda x: int(x[:4]))

    def analyze_state_trends(self, metric: str = "sse_pc") -> dict:
        """
        Computes growth rate, log-linear trend, Mann-Kendall, and Sen's slope for each state.
        """
        results = {}
        for state_code, state_rows in self.by_state.items():
            vals = [float(r.get(metric, 0) or 0) for r in state_rows]
            yrs = [int(r["year_t"]) for r in state_rows]
            
            # Full period CAGR
            cagr_full = calculate_cagr(vals[0], vals[-1], len(vals) - 1)
            
            # Sub-periods CAGR:
            # Pre-GST: 2013-14 to 2016-17 (idx 0 to 3)
            cagr_pre_gst = calculate_cagr(vals[0], vals[3], 3)
            # GST to Pre-COVID: 2016-17 to 2019-20 (idx 3 to 6)
            cagr_gst_precovid = calculate_cagr(vals[3], vals[6], 3)
            # COVID: 2019-20 to 2021-22 (idx 6 to 8)
            cagr_covid = calculate_cagr(vals[6], vals[8], 2)
            # Post-COVID: 2021-22 to 2025-26 (idx 8 to 12)
            cagr_post_covid = calculate_cagr(vals[8], vals[-1], len(vals) - 1 - 8)
            
            ll_res = log_linear_trend(vals, yrs)
            mk_res = mann_kendall_test(vals)
            s_slope = sens_slope(vals)
            
            results[state_code] = {
                "state_name": state_rows[0]["state_name"],
                "category": state_rows[0]["category"],
                "initial_val": vals[0],
                "final_val": vals[-1],
                "cagr_full_pct": round(cagr_full * 100.0, 2) if not np.isnan(cagr_full) else None,
                "cagr_pre_gst_pct": round(cagr_pre_gst * 100.0, 2) if not np.isnan(cagr_pre_gst) else None,
                "cagr_gst_precovid_pct": round(cagr_gst_precovid * 100.0, 2) if not np.isnan(cagr_gst_precovid) else None,
                "cagr_covid_pct": round(cagr_covid * 100.0, 2) if not np.isnan(cagr_covid) else None,
                "cagr_post_covid_pct": round(cagr_post_covid * 100.0, 2) if not np.isnan(cagr_post_covid) else None,
                "log_linear_growth_pct": round(ll_res["growth_rate"] * 100.0, 2) if not np.isnan(ll_res["growth_rate"]) else None,
                "log_linear_r2": round(ll_res["r2"], 4) if not np.isnan(ll_res["r2"]) else None,
                "log_linear_p_value": round(ll_res["p_value"], 5) if not np.isnan(ll_res["p_value"]) else None,
                "mann_kendall_trend": mk_res["trend"],
                "mann_kendall_tau": round(mk_res["tau"], 3) if not np.isnan(mk_res["tau"]) else None,
                "mann_kendall_p": round(mk_res["p_value"], 5) if not np.isnan(mk_res["p_value"]) else None,
                "sens_slope": round(s_slope, 2) if not np.isnan(s_slope) else None
            }
        return results

    def analyze_convergence(self, metric: str = "sse_pc") -> dict:
        """
        Computes Sigma-convergence (CV by year) and Beta-convergence (unconditional).
        """
        # 1. Sigma convergence
        by_year = defaultdict(list)
        for r in self.records:
            by_year[r["fiscal_year"]].append(float(r.get(metric, 0) or 0))
            
        sigma_res = sigma_convergence(by_year)
        
        # 2. Beta convergence (Full period: 2013-14 to 2025-26)
        initial_vals = []
        final_vals = []
        for s_code, rows in self.by_state.items():
            initial_vals.append(float(rows[0].get(metric, 0) or 0))
            final_vals.append(float(rows[-1].get(metric, 0) or 0))
            
        periods = len(self.years) - 1
        beta_res = beta_convergence(initial_vals, final_vals, periods)
        
        # Also compute for core audited period (2013-14 to 2023-24) as robustness check
        init_core = []
        final_core = []
        for s_code, rows in self.by_state.items():
            init_core.append(float(rows[0].get(metric, 0) or 0))
            final_core.append(float(rows[10].get(metric, 0) or 0)) # 10 is 2023-24
        beta_core = beta_convergence(init_core, final_core, 10)
        
        return {
            "metric": metric,
            "sigma_convergence_by_year": sigma_res,
            "beta_convergence_full": beta_res,
            "beta_convergence_audited_only": beta_core
        }

    def analyze_composition_shift(self) -> dict:
        """
        Measures structural shifts in sub-sector expenditure shares across periods.
        """
        # Calculate national average sub-sector shares for baseline (2013-14), COVID (2020-21), and terminal (2025-26)
        sub_sectors = ["edu", "hea", "wsh", "ssw", "nut", "scst"]
        
        def avg_shares_for_year(target_year):
            shares = {sub: [] for sub in sub_sectors}
            for r in self.records:
                if r["fiscal_year"] == target_year:
                    for sub in sub_sectors:
                        shares[sub].append(float(r.get(f"{sub}_share_sse", 0) or 0))
            return {sub: round(float(np.mean(shares[sub])), 2) for sub in sub_sectors}
            
        baseline_shares = avg_shares_for_year("2013-14")
        covid_shares = avg_shares_for_year("2020-21")
        terminal_shares = avg_shares_for_year("2025-26")
        
        d_baseline_covid = compute_dissimilarity_index(
            list(baseline_shares.values()), list(covid_shares.values())
        )
        d_covid_terminal = compute_dissimilarity_index(
            list(covid_shares.values()), list(terminal_shares.values())
        )
        d_baseline_terminal = compute_dissimilarity_index(
            list(baseline_shares.values()), list(terminal_shares.values())
        )
        
        return {
            "sub_sectors": sub_sectors,
            "shares_2013_14": baseline_shares,
            "shares_2020_21_covid": covid_shares,
            "shares_2025_26_terminal": terminal_shares,
            "dissimilarity_index_covid_shock": round(d_baseline_covid, 4),
            "dissimilarity_index_postcovid_recovery": round(d_covid_terminal, 4),
            "dissimilarity_index_overall_structural_shift": round(d_baseline_terminal, 4)
        }
