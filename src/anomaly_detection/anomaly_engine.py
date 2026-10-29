"""
Social Spending Anomaly Detection Engine
Implements statistical MAD robust z-scores, multivariate Isolation Forest,
Local Outlier Factor (LOF), and audit triangulation with external shocks.
"""

import csv
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.utils.math_stats import robust_z_scores


class AnomalyDetectionEngine:
    def __init__(self, panel_csv_path: str):
        self.records = []
        with open(panel_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.records.append(r)
                
        # Group by state to compute YoY growth rates
        self.by_state = defaultdict(list)
        for r in self.records:
            self.by_state[r["state_code"]].append(r)
            
        for s in self.by_state:
            self.by_state[s].sort(key=lambda x: int(x["year_t"]))
            
        self._compute_growth_metrics()

    def _compute_growth_metrics(self):
        for s, rows in self.by_state.items():
            for i, r in enumerate(rows):
                if i == 0:
                    r["sse_real_yoy_growth"] = 0.0
                    r["hea_share_yoy_diff"] = 0.0
                else:
                    prev = rows[i - 1]
                    prev_val = float(prev["sse_real"])
                    curr_val = float(r["sse_real"])
                    growth = ((curr_val - prev_val) / prev_val) * 100.0 if prev_val > 0 else 0.0
                    r["sse_real_yoy_growth"] = round(growth, 2)
                    
                    prev_hea = float(prev["hea_share_sse"])
                    curr_hea = float(r["hea_share_sse"])
                    r["hea_share_yoy_diff"] = round(curr_hea - prev_hea, 2)

    def detect_statistical_anomalies(self, z_threshold: float = 2.8) -> list:
        """
        Flags state-years with extreme year-on-year growth rate using Robust Z-Score (MAD).
        """
        # Exclude initial year 2013-14 (since YoY growth is 0)
        eval_records = [r for r in self.records if r["fiscal_year"] != "2013-14"]
        growths = [float(r["sse_real_yoy_growth"]) for r in eval_records]
        z_scores = robust_z_scores(growths)
        
        flags = []
        for i, r in enumerate(eval_records):
            z = z_scores[i]
            if abs(z) >= z_threshold:
                flags.append({
                    "state_code": r["state_code"],
                    "state_name": r["state_name"],
                    "fiscal_year": r["fiscal_year"],
                    "metric": "sse_real_yoy_growth",
                    "value": float(r["sse_real_yoy_growth"]),
                    "robust_z_score": round(float(z), 2),
                    "anomaly_type": "Statistical (MAD Z-Score)"
                })
        return flags

    def detect_multivariate_anomalies(self, contamination: float = 0.04) -> list:
        """
        Flags multivariate outliers using Isolation Forest and Local Outlier Factor.
        """
        feature_keys = ["sse_pc", "sse_gsdp", "sse_te", "hea_share_sse", "be_act", "sav_exc"]
        X = []
        for r in self.records:
            vec = [float(r.get(k, 0) or 0) for k in feature_keys]
            X.append(vec)
        X = np.array(X)
        
        # Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso_preds = iso.fit_predict(X) # -1 for anomaly, 1 for inlier
        iso_scores = iso.decision_function(X)
        
        # LOF
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_preds = lof.fit_predict(X)
        lof_factors = lof.negative_outlier_factor_
        
        flags = []
        for i, r in enumerate(self.records):
            is_iso_outlier = (iso_preds[i] == -1)
            is_lof_outlier = (lof_preds[i] == -1)
            
            # If flagged by either model
            if is_iso_outlier or is_lof_outlier:
                flags.append({
                    "state_code": r["state_code"],
                    "state_name": r["state_name"],
                    "fiscal_year": r["fiscal_year"],
                    "iso_forest_outlier": bool(is_iso_outlier),
                    "iso_score": round(float(iso_scores[i]), 3),
                    "lof_outlier": bool(is_lof_outlier),
                    "lof_factor": round(float(lof_factors[i]), 3),
                    "be_act": float(r["be_act"]),
                    "sav_exc": float(r["sav_exc"]),
                    "sse_pc": float(r["sse_pc"]),
                    "anomaly_type": "Multivariate ML"
                })
        return flags

    def detect_budget_execution_anomalies(self, be_low: float = 0.82, be_high: float = 1.08) -> list:
        """
        Flags severe under-execution or large excess/savings over legislative grants.
        """
        flags = []
        for r in self.records:
            be = float(r["be_act"])
            sav = float(r["sav_exc"])
            if be < be_low or be > be_high or abs(sav) > 15.0:
                flags.append({
                    "state_code": r["state_code"],
                    "state_name": r["state_name"],
                    "fiscal_year": r["fiscal_year"],
                    "be_act": be,
                    "sav_exc_pct": sav,
                    "anomaly_type": "Budget Execution Divergence"
                })
        return flags

    def triangulate_anomalies(self) -> list:
        """
        Compiles and triangulates all detected anomalies with known fiscal and institutional events.
        Produces the formal Anomaly Register.
        """
        stat_flags = {(f["state_code"], f["fiscal_year"]): f for f in self.detect_statistical_anomalies()}
        ml_flags = {(f["state_code"], f["fiscal_year"]): f for f in self.detect_multivariate_anomalies()}
        be_flags = {(f["state_code"], f["fiscal_year"]): f for f in self.detect_budget_execution_anomalies()}
        
        all_keys = sorted(list(set(stat_flags.keys()) | set(ml_flags.keys()) | set(be_flags.keys())))
        
        register = []
        for state_code, fy in all_keys:
            r = next((x for x in self.records if x["state_code"] == state_code and x["fiscal_year"] == fy), None)
            if not r:
                continue
                
            methods = []
            if (state_code, fy) in stat_flags:
                methods.append("Statistical (MAD)")
            if (state_code, fy) in ml_flags:
                methods.append("Machine Learning (IsoForest/LOF)")
            if (state_code, fy) in be_flags:
                methods.append("Budget Execution Shortfall/Excess")
                
            # Triangulation reasoning
            category = "C: Statistical Outlier (CAG Audit Follow-up)"
            explanation = "Unusual departure from peer trajectory without immediate macro explanation."
            recommendation = "Review Detailed Appropriation Accounts and scheme-wise surrender orders."
            
            # Check known events:
            if fy in ("2020-21", "2021-22"):
                category = "A: Event-Explained (COVID-19 Pandemic)"
                explanation = "National emergency reallocation: Health and social relief expenditure surged sharply while urban/capital works stalled."
                recommendation = "Reconcile emergency procurement accounts against State Disaster Response Fund (SDRF) / PM-CARES allocations."
            elif state_code == "AP" and fy in ("2014-15", "2015-16"):
                category = "A: Event-Explained (State Reorganisation)"
                explanation = "Andhra Pradesh Reorganisation Act 2014 transitional restructuring, asset/liability apportionment and administrative bifurcation."
                recommendation = "Verify apportionment schedules in Combined Finance Accounts and AG reconciliation certificates."
            elif state_code == "JK" and fy in ("2019-20", "2020-21"):
                category = "A: Event-Explained (J&K Reorganisation)"
                explanation = "Bifurcation of Jammu & Kashmir into UT of J&K and UT of Ladakh under J&K Reorganisation Act 2019."
                recommendation = "Inspect opening balance adjustments in UT Accounts and transitional grants from Union Home Ministry."
            elif state_code == "KL" and fy == "2018-19":
                category = "A: Event-Explained (Natural Calamity - Kerala Floods)"
                explanation = "Century flood disaster necessitated massive immediate relief and emergency rehabilitation spending under Head 2245."
                recommendation = "Audit State Disaster Response Fund (SDRF) drawdown and emergency reconstruction appropriations."
            elif float(r["be_act"]) < 0.82:
                category = "B: Execution Bottleneck (Severe Underspending)"
                explanation = f"Significant shortfall against budget estimate (Execution rate {float(r['be_act'])*100:.1f}%); persistent unspent balances."
                recommendation = "Examine procedural delays in administrative sanctions, delays in central release matching shares, and quarterly lapse patterns."
            elif float(r["be_act"]) > 1.08 or float(r["sav_exc"]) > 10.0:
                category = "B: Excess Expenditure / Supplementary Grant Surge"
                explanation = f"Expenditure exceeded original legislative grant by {float(r['sav_exc']):.1f}%, requiring regularisation."
                recommendation = "Verify whether excess expenditure was covered by approved supplementary demands or unratified re-appropriations under Article 205."

            register.append({
                "state_code": state_code,
                "state_name": r["state_name"],
                "fiscal_year": fy,
                "detection_methods": ", ".join(methods),
                "triangulation_category": category,
                "context_explanation": explanation,
                "audit_recommendation": recommendation,
                "metrics_summary": {
                    "sse_pc": float(r["sse_pc"]),
                    "sse_gsdp": float(r["sse_gsdp"]),
                    "be_act": float(r["be_act"]),
                    "sav_exc_pct": float(r["sav_exc"]),
                    "hea_share": float(r["hea_share_sse"])
                }
            })
            
        return register
