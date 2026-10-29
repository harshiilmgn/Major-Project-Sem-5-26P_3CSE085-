"""
Data Validation and Quality Control
Implements accounting identity reconciliation, tolerance limits, and an automated audit log.
"""

from typing import List, Dict, Any, Tuple


class DataValidator:
    def __init__(self, tolerance_pct: float = 2.0):
        self.tolerance_pct = tolerance_pct

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates an individual state-year observation against accounting rules.
        """
        issues = []
        state = record.get("state_code", "UNKNOWN")
        year = record.get("fiscal_year", "UNKNOWN")
        
        sse = float(record.get("sse", 0.0) or 0.0)
        te = float(record.get("total_exp", 0.0) or 0.0)
        
        # 1. Total Expenditure check
        if te > 0 and sse > te:
            issues.append(f"CRITICAL: SSE ({sse}) exceeds Total Expenditure ({te})")
            
        # 2. Sub-sector sum check
        sub_sum = sum(float(record.get(sub, 0.0) or 0.0) for sub in ["edu", "hea", "wsh", "ssw", "nut", "scst", "other_social"])
        if sse > 0 and sub_sum > 0:
            diff_pct = abs(sub_sum - sse) / sse * 100.0
            if diff_pct > self.tolerance_pct:
                issues.append(f"WARNING: Sub-sectors sum ({sub_sum:.2f}) diverges from SSE ({sse:.2f}) by {diff_pct:.2f}% (tolerance {self.tolerance_pct}%)")
                
        # 3. Budget execution sanity check
        be_act = record.get("be_act")
        if be_act is not None:
            be_act_val = float(be_act)
            if be_act_val < 0.4 or be_act_val > 2.5:
                issues.append(f"FLAG: Budget Execution Ratio BE_ACT ({be_act_val:.2f}) indicates severe under/over-spending")
                
        # 4. GSDP plausibility
        gsdp = float(record.get("gsdp_nominal", 0.0) or 0.0)
        if gsdp > 0 and sse > 0:
            ratio = (sse / gsdp) * 100.0
            if ratio > 35.0:
                issues.append(f"FLAG: Social spending as % GSDP is extraordinarily high ({ratio:.2f}%)")
                
        is_valid = len([i for i in issues if i.startswith("CRITICAL")]) == 0
        return is_valid, issues

    def audit_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs validation across all records and returns an audit log summary.
        """
        audit_log = []
        valid_count = 0
        flag_count = 0
        
        for r in records:
            valid, issues = self.validate_record(r)
            if valid:
                valid_count += 1
            if issues:
                flag_count += 1
                audit_log.append({
                    "state_code": r.get("state_code"),
                    "fiscal_year": r.get("fiscal_year"),
                    "issues": issues
                })
                
        return {
            "total_records": len(records),
            "valid_records": valid_count,
            "records_with_flags": flag_count,
            "pass_rate_pct": round(valid_count / len(records) * 100.0, 2) if records else 0,
            "log": audit_log
        }
