"""
Unit and Integration Tests for CAG-ICSSR Research Pipeline
"""

import os
import sys
import pytest
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.utils.math_stats import (
    calculate_cagr,
    log_linear_trend,
    mann_kendall_test,
    sens_slope,
    robust_z_scores,
    compute_dissimilarity_index
)
from src.data_processing.validator import DataValidator
from src.trend_analysis.trend_engine import TrendAnalysisEngine
from src.clustering.cluster_engine import ClusterAnalysisEngine
from src.anomaly_detection.anomaly_engine import AnomalyDetectionEngine
from src.econometrics.panel_regressor import PanelRegressionEngine


def test_calculate_cagr():
    # 100 to 200 in 10 years: (200/100)^(0.1) - 1 = 2^0.1 - 1 ~= 0.07177
    cagr = calculate_cagr(100.0, 200.0, 10)
    assert abs(cagr - 0.07177) < 1e-4
    assert np.isnan(calculate_cagr(0, 100, 5))
    assert np.isnan(calculate_cagr(100, -10, 5))


def test_mann_kendall_monotonic_trend():
    # Strictly increasing sequence
    seq_inc = [10, 12, 15, 18, 22, 25, 30]
    res_inc = mann_kendall_test(seq_inc)
    assert res_inc["trend"] == "increasing"
    assert res_inc["tau"] == 1.0
    assert res_inc["p_value"] < 0.05

    # Strictly decreasing sequence
    seq_dec = [50, 42, 38, 29, 21, 15]
    res_dec = mann_kendall_test(seq_dec)
    assert res_dec["trend"] == "decreasing"
    assert res_dec["tau"] == -1.0
    assert res_dec["p_value"] < 0.05


def test_sens_slope():
    x = [2, 4, 6, 8, 10]
    assert sens_slope(x) == 2.0


def test_robust_z_score():
    vals = [10, 11, 10, 12, 10, 100]
    z = robust_z_scores(vals)
    # The outlier (100) should have a huge robust z-score > 3
    assert z[-1] > 3.0


def test_dissimilarity_index():
    s1 = [0.5, 0.5]
    s2 = [0.5, 0.5]
    assert compute_dissimilarity_index(s1, s2) == 0.0
    s3 = [1.0, 0.0]
    s4 = [0.0, 1.0]
    assert compute_dissimilarity_index(s3, s4) == 1.0


def test_validator():
    val = DataValidator(tolerance_pct=2.0)
    # Valid record
    rec_ok = {
        "state_code": "TN",
        "fiscal_year": "2020-21",
        "total_exp": 1000.0,
        "sse": 400.0,
        "edu": 160.0, "hea": 80.0, "wsh": 60.0, "ssw": 60.0, "nut": 20.0, "scst": 10.0, "other_social": 10.0,
        "be_act": 0.95,
        "gsdp_nominal": 5000.0
    }
    is_valid, issues = val.validate_record(rec_ok)
    assert is_valid is True
    assert len(issues) == 0

    # Inconsistent sum
    rec_bad = dict(rec_ok)
    rec_bad["sse"] = 500.0 # diverges from sum 400
    is_valid_bad, issues_bad = val.validate_record(rec_bad)
    assert any("diverges" in i for i in issues_bad)


def test_panel_pipeline_execution():
    panel_path = os.path.join(WORKSPACE_DIR, "data", "processed", "panel_2013_2026.csv")
    assert os.path.exists(panel_path)
    
    # Check trend engine runs
    te = TrendAnalysisEngine(panel_path)
    trends = te.analyze_state_trends()
    assert "MH" in trends
    
    # Check clustering runs
    ce = ClusterAnalysisEngine(panel_path)
    cl = ce.run_clustering(n_clusters=3, algorithm="kmeans")
    assert cl["n_clusters"] == 3
    
    # Check econometrics runs
    pe = PanelRegressionEngine(panel_path)
    m = pe.estimate_model(dep_var="ln_sse_pc")
    assert "within_r2" in m
    assert m["hausman_test"]["preferred_model"] in ("Fixed Effects (FE)", "Random Effects (RE)")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
