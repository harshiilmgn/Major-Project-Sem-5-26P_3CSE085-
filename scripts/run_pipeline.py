"""
End-to-End Pipeline Runner
Executes the full computational pipeline: validation, trend/convergence,
clustering, anomaly detection & triangulation, and panel econometrics.
Outputs structured JSON and Markdown artifacts.
"""

import os
import sys
import json
import csv

# Add workspace to path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from src.data_processing.validator import DataValidator
from src.trend_analysis.trend_engine import TrendAnalysisEngine
from src.clustering.cluster_engine import ClusterAnalysisEngine
from src.anomaly_detection.anomaly_engine import AnomalyDetectionEngine
from src.econometrics.panel_regressor import PanelRegressionEngine

PANEL_PATH = os.path.join(WORKSPACE_DIR, "data", "processed", "panel_2013_2026.csv")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "reports")


def run_full_pipeline():
    print("=" * 80)
    print("CAG-ICSSR COMPUTATIONAL RESEARCH PIPELINE: SOCIAL SECTOR SPENDING")
    print("=" * 80)

    # 1. Validation
    print("\n[Phase 1] Auditing and Validating Panel Dataset...")
    records = []
    with open(PANEL_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            
    validator = DataValidator()
    audit_summary = validator.audit_dataset(records)
    print(f"  Total records: {audit_summary['total_records']}")
    print(f"  Valid records: {audit_summary['valid_records']} ({audit_summary['pass_rate_pct']}%)")
    print(f"  Flagged issues: {audit_summary['records_with_flags']}")

    # 2. Trend & Convergence Analysis
    print("\n[Phase 2] Computing Trends, Mann-Kendall, and Convergence Dynamics...")
    trend_engine = TrendAnalysisEngine(PANEL_PATH)
    trends = trend_engine.analyze_state_trends("sse_pc")
    conv = trend_engine.analyze_convergence("sse_pc")
    comp_shift = trend_engine.analyze_composition_shift()
    
    print(f"  Beta Convergence (Full Period): Beta = {conv['beta_convergence_full']['beta']:.4f}, p = {conv['beta_convergence_full']['p_value']:.4f}")
    print(f"  Beta Convergence (Audited Only): Beta = {conv['beta_convergence_audited_only']['beta']:.4f}, p = {conv['beta_convergence_audited_only']['p_value']:.4f}")
    print(f"  Dissimilarity Index (COVID Shock): {comp_shift['dissimilarity_index_covid_shock']}")

    # 3. Cluster Analysis
    print("\n[Phase 3] Unsupervised State Clustering & Validation...")
    cluster_engine = ClusterAnalysisEngine(PANEL_PATH)
    cluster_eval = cluster_engine.evaluate_cluster_numbers(k_min=2, k_max=6)
    print("  Cluster Evaluation Metrics:")
    for m in cluster_eval:
        print(f"    k={m['k']}: Silhouette={m['silhouette_score']}, Calinski-Harabasz={m['calinski_harabasz_score']}, Davies-Bouldin={m['davies_bouldin_score']}")
        
    best_k = max(cluster_eval, key=lambda x: x["silhouette_score"])["k"]
    cluster_res = cluster_engine.run_clustering(n_clusters=best_k, algorithm="kmeans")
    print(f"  Selected k={best_k} (Silhouette = {cluster_res['silhouette_score']})")

    # 4. Anomaly Detection & Triangulation
    print("\n[Phase 4] Anomaly Detection & Audit Triangulation...")
    anomaly_engine = AnomalyDetectionEngine(PANEL_PATH)
    register = anomaly_engine.triangulate_anomalies()
    print(f"  Total Triangulated Anomalies in Register: {len(register)}")
    cat_counts = {}
    for entry in register:
        cat = entry["triangulation_category"].split(":")[0]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"    Category {cat}: {count} observations")

    # 5. Econometric Panel Models
    print("\n[Phase 5] Econometric Panel Regressions (Determinants of Social Spending)...")
    reg_engine = PanelRegressionEngine(PANEL_PATH)
    model_sse_pc = reg_engine.estimate_model(dep_var="ln_sse_pc")
    model_sse_te = reg_engine.estimate_model(dep_var="sse_te")
    
    print(f"  Model 1: ln(SSE_PC)")
    print(f"    Within R2: {model_sse_pc['within_r2']}")
    print(f"    Hausman Test: Chi2 = {model_sse_pc['hausman_test']['chi2_stat']}, p = {model_sse_pc['hausman_test']['p_value']} -> Preferred: {model_sse_pc['hausman_test']['preferred_model']}")
    print("    Fixed Effects Coefficients:")
    for coef_info in model_sse_pc["fixed_effects_results"]:
        print(f"      {coef_info['variable']:<15}: {coef_info['coef']:>8.4f} (SE: {coef_info['std_err']:.4f}, p: {coef_info['p_value']:.4f}) {coef_info['significance']}")

    # 6. Save Pipeline Output Artifacts
    output_bundle = {
        "audit": audit_summary,
        "trends_sample": {k: trends[k] for k in list(trends.keys())[:5]},
        "convergence": conv,
        "composition_shifts": comp_shift,
        "cluster_evaluation": cluster_eval,
        "clusters": cluster_res,
        "anomaly_register_count": len(register),
        "anomaly_register": register,
        "econometric_models": {
            "ln_sse_pc": model_sse_pc,
            "sse_te": model_sse_te
        }
    }
    
    out_json = os.path.join(REPORTS_DIR, "tables", "pipeline_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_bundle, f, indent=2)
    print(f"\n[Artifact] Pipeline JSON exported to: {out_json}")
    
    # Save CSV Anomaly Register
    out_anom_csv = os.path.join(REPORTS_DIR, "anomaly_register", "anomaly_register.csv")
    os.makedirs(os.path.dirname(out_anom_csv), exist_ok=True)
    with open(out_anom_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["state_code", "state_name", "fiscal_year", "detection_methods", "triangulation_category", "context_explanation", "audit_recommendation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in register:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"[Artifact] Anomaly Register CSV exported to: {out_anom_csv}")
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
