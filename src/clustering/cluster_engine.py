"""
State Expenditure Clustering Engine
Implements K-Means, Agglomerative Hierarchical (Ward), and Gaussian Mixture Models.
Evaluates cluster stability using Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
"""

import csv
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats


class ClusterAnalysisEngine:
    def __init__(self, panel_csv_path: str, exclude_provisional: bool = False, drop_jk: bool = False):
        self.records = []
        with open(panel_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if exclude_provisional and int(r["is_provisional"]) == 1:
                    continue
                if drop_jk and r["state_code"] == "JK":
                    continue
                self.records.append(r)
                
        # Group by state to build level features
        self.by_state = defaultdict(list)
        for r in self.records:
            self.by_state[r["state_code"]].append(r)
            
        self.state_codes = sorted(list(self.by_state.keys()))
        self.feature_names = [
            "sse_pc_mean",
            "sse_gsdp_mean",
            "sse_te_mean",
            "edu_share_mean",
            "hea_share_mean",
            "wsh_share_mean",
            "otr_gsdp_mean",
            "dev_rr_mean",
            "be_act_mean"
        ]
        self.feature_matrix, self.state_meta = self._build_feature_matrix()
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.feature_matrix)

    def _build_feature_matrix(self):
        matrix = []
        meta = []
        for code in self.state_codes:
            rows = self.by_state[code]
            meta.append({
                "state_code": code,
                "state_name": rows[0]["state_name"],
                "category": rows[0]["category"],
                "region": rows[0]["region"]
            })
            
            vec = [
                float(np.mean([float(r["sse_pc"]) for r in rows])),
                float(np.mean([float(r["sse_gsdp"]) for r in rows])),
                float(np.mean([float(r["sse_te"]) for r in rows])),
                float(np.mean([float(r["edu_share_sse"]) for r in rows])),
                float(np.mean([float(r["hea_share_sse"]) for r in rows])),
                float(np.mean([float(r["wsh_share_sse"]) for r in rows])),
                float(np.mean([float(r["otr_gsdp"]) for r in rows])),
                float(np.mean([float(r["dev_rr"]) for r in rows])),
                float(np.mean([float(r["be_act"]) for r in rows]))
            ]
            matrix.append(vec)
        return np.array(matrix), meta

    def evaluate_cluster_numbers(self, k_min: int = 2, k_max: int = 6) -> dict:
        """
        Calculates Silhouette, Calinski-Harabasz, and Davies-Bouldin across k.
        """
        metrics = []
        for k in range(k_min, min(k_max + 1, len(self.state_codes))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.scaled_features)
            
            sil = silhouette_score(self.scaled_features, labels)
            ch = calinski_harabasz_score(self.scaled_features, labels)
            db = davies_bouldin_score(self.scaled_features, labels)
            
            metrics.append({
                "k": k,
                "silhouette_score": round(float(sil), 4),
                "calinski_harabasz_score": round(float(ch), 2),
                "davies_bouldin_score": round(float(db), 4)
            })
        return metrics

    def run_clustering(self, n_clusters: int = 4, algorithm: str = "kmeans") -> dict:
        """
        Runs clustering and returns assignments and profile statistics.
        algorithm: 'kmeans', 'hierarchical', 'gmm'
        """
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(self.scaled_features)
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(self.scaled_features)
        elif algorithm == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(self.scaled_features)
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")
            
        sil = float(silhouette_score(self.scaled_features, labels))
        
        # Build cluster mapping
        clusters = defaultdict(list)
        for i, code in enumerate(self.state_codes):
            c_id = int(labels[i])
            clusters[c_id].append({
                "state_code": code,
                "state_name": self.state_meta[i]["state_name"],
                "category": self.state_meta[i]["category"],
                "region": self.state_meta[i]["region"],
                "features": {name: round(float(self.feature_matrix[i, j]), 2) for j, name in enumerate(self.feature_names)}
            })
            
        # Profile each cluster
        cluster_profiles = {}
        for c_id, members in clusters.items():
            profile = {
                "size": len(members),
                "states": [m["state_name"] for m in members],
                "categories": [m["category"] for m in members],
                "mean_metrics": {}
            }
            for j, name in enumerate(self.feature_names):
                vals = [m["features"][name] for m in members]
                profile["mean_metrics"][name] = round(float(np.mean(vals)), 2)
                
            cluster_profiles[c_id] = profile
            
        # Kruskal-Wallis test across clusters for each feature
        kw_tests = {}
        for j, name in enumerate(self.feature_names):
            groups = []
            for c_id in sorted(clusters.keys()):
                vals = [m["features"][name] for m in clusters[c_id]]
                groups.append(vals)
            stat, p_val = stats.kruskal(*groups)
            kw_tests[name] = {
                "h_stat": round(float(stat), 3),
                "p_value": round(float(p_val), 5),
                "significant": bool(p_val < 0.05)
            }
            
        return {
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            "silhouette_score": round(sil, 4),
            "clusters": cluster_profiles,
            "kruskal_wallis_tests": kw_tests
        }
