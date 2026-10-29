"""
State Finance Data Harmoniser
Handles historical breaks (AP/Telangana 2014, J&K 2019), deflates nominal figures,
and computes real per capita metrics.
"""

import json
import csv
from typing import List, Dict, Any, Optional


class StateDataHarmonizer:
    def __init__(self, states_config_path: str, deflators_path: str):
        with open(states_config_path, "r", encoding="utf-8") as f:
            self.states_config = {s["code"]: s for s in json.load(f)}
            
        self.deflators = {}
        with open(deflators_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.deflators[row["fiscal_year"]] = {
                    "gsdp_deflator": float(row["gsdp_deflator"]),
                    "cpi_index": float(row["cpi_index"]),
                    "is_provisional": int(row["is_provisional"])
                }

    def harmonize_ap_telangana(self, records: List[Dict[str, Any]], method: str = "apportion") -> List[Dict[str, Any]]:
        """
        Harmonises 2013-14 for Andhra Pradesh and Telangana.
        method='apportion': splits combined AP 2013-14 using AP Reorganisation Act population ratio (58.32% AP, 41.68% TS).
        method='combined': keeps AP and TS combined as a single unit ('AP_COMBINED') across all years.
        """
        if method not in ("apportion", "combined"):
            raise ValueError("Method must be 'apportion' or 'combined'")
            
        output = []
        if method == "apportion":
            for r in records:
                if r["state_code"] == "AP_COMBINED" and r["fiscal_year"] == "2013-14":
                    ap_ratio = 0.5832
                    ts_ratio = 0.4168
                    
                    row_ap = dict(r)
                    row_ap["state_code"] = "AP"
                    row_ap["state_name"] = "Andhra Pradesh"
                    row_ap["notes"] = "Apportioned 58.32% from undivided AP"
                    
                    row_ts = dict(r)
                    row_ts["state_code"] = "TS"
                    row_ts["state_name"] = "Telangana"
                    row_ts["notes"] = "Apportioned 41.68% from undivided AP"
                    
                    metric_keys = ["sse", "edu", "hea", "wsh", "ssw", "nut", "scst", "other_social", "total_exp", "gsdp_nominal", "population"]
                    for k in metric_keys:
                        if k in r and r[k] is not None:
                            row_ap[k] = round(float(r[k]) * ap_ratio, 2)
                            row_ts[k] = round(float(r[k]) * ts_ratio, 2)
                            
                    output.extend([row_ap, row_ts])
                else:
                    output.append(r)
        else: # method == "combined"
            by_year = {}
            for r in records:
                yr = r["fiscal_year"]
                code = r["state_code"]
                if code in ("AP", "TS", "AP_COMBINED"):
                    if yr not in by_year:
                        by_year[yr] = []
                    by_year[yr].append(r)
                else:
                    output.append(r)
                    
            metric_keys = ["sse", "edu", "hea", "wsh", "ssw", "nut", "scst", "other_social", "total_exp", "gsdp_nominal", "population"]
            for yr, rows in by_year.items():
                if len(rows) == 1 and rows[0]["state_code"] == "AP_COMBINED":
                    output.append(rows[0])
                else:
                    comb = dict(rows[0])
                    comb["state_code"] = "AP_COMBINED"
                    comb["state_name"] = "Andhra Pradesh (Combined)"
                    for k in metric_keys:
                        comb[k] = sum(float(x.get(k, 0) or 0) for x in rows)
                    comb["notes"] = "Combined AP + Telangana for post-2014 comparability"
                    output.append(comb)
                    
        return output

    def filter_jk(self, records: List[Dict[str, Any]], drop_jk: bool = False) -> List[Dict[str, Any]]:
        """
        Handles Jammu & Kashmir 2019 structural break.
        If drop_jk=True, excludes J&K from the sample.
        If drop_jk=False, marks post-2019 observations with structural_break flag.
        """
        output = []
        for r in records:
            if r["state_code"] == "JK":
                if drop_jk:
                    continue
                r = dict(r)
                year_num = int(r["fiscal_year"][:4])
                if year_num >= 2019:
                    r["is_structural_break"] = 1
                    r["notes"] = (r.get("notes", "") + " Post-reorganisation UT data").strip()
                else:
                    r["is_structural_break"] = 0
            else:
                r["is_structural_break"] = 0
            output.append(r)
        return output

    def deflate_and_normalize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts nominal figures to real (2011-12 prices) and computes per capita & share ratios.
        """
        rec = dict(record)
        fy = rec["fiscal_year"]
        defl_info = self.deflators.get(fy, {"gsdp_deflator": 100.0, "is_provisional": 0})
        deflator = defl_info["gsdp_deflator"]
        
        rec["is_provisional"] = defl_info["is_provisional"]
        rec["gsdp_deflator"] = deflator
        
        # Real expenditure (in Rs crore at 2011-12 prices)
        sse_nom = float(rec.get("sse", 0.0) or 0.0)
        rec["sse_real"] = round(sse_nom * 100.0 / deflator, 2)
        
        for sub in ["edu", "hea", "wsh", "ssw", "nut", "scst"]:
            sub_nom = float(rec.get(sub, 0.0) or 0.0)
            rec[f"{sub}_real"] = round(sub_nom * 100.0 / deflator, 2)
            
        pop = float(rec.get("population", 0.0) or 0.0)
        if pop > 0:
            rec["sse_pc"] = round((rec["sse_real"] * 1e7) / pop, 2)
            for sub in ["edu", "hea", "wsh", "ssw", "nut", "scst"]:
                rec[f"{sub}_pc"] = round((rec[f"{sub}_real"] * 1e7) / pop, 2)
        else:
            rec["sse_pc"] = None
            for sub in ["edu", "hea", "wsh", "ssw", "nut", "scst"]:
                rec[f"{sub}_pc"] = None
                
        gsdp = float(rec.get("gsdp_nominal", 0.0) or 0.0)
        te = float(rec.get("total_exp", 0.0) or 0.0)
        
        rec["sse_gsdp"] = round((sse_nom / gsdp) * 100.0, 2) if gsdp > 0 else None
        rec["sse_te"] = round((sse_nom / te) * 100.0, 2) if te > 0 else None
        
        for sub in ["edu", "hea", "wsh", "ssw", "nut", "scst"]:
            sub_nom = float(rec.get(sub, 0.0) or 0.0)
            rec[f"{sub}_share_sse"] = round((sub_nom / sse_nom) * 100.0, 2) if sse_nom > 0 else None
            rec[f"{sub}_share_te"] = round((sub_nom / te) * 100.0, 2) if te > 0 else None
            
        fy_year = int(fy[:4])
        if fy_year < 2015:
            rec["fc_period"] = "13th_FC"
        elif 2015 <= fy_year < 2020:
            rec["fc_period"] = "14th_FC"
        else:
            rec["fc_period"] = "15th_FC"
            
        rec["covid_shock"] = 1 if fy in ("2020-21", "2021-22") else 0
        
        state_meta = self.states_config.get(rec["state_code"], {})
        rec["category"] = state_meta.get("category", "GCS")
        rec["region"] = state_meta.get("region", "Other")
        rec["is_special_category"] = 1 if rec["category"] == "SCS" else 0
        
        return rec
