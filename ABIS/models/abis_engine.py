"""
ABIS Engine - Core AI/ML Pipeline
Adaptive Behavioral Intelligence System for Resource Utilization Optimization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    """Generates realistic synthetic system resource data"""

    @staticmethod
    def generate(n_samples=2000, seed=42):
        np.random.seed(seed)
        hours = np.arange(n_samples) % 24
        days  = np.arange(n_samples) // 24

        # CPU usage with daily patterns
        base_cpu = (
            30 + 40 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
            + np.random.normal(0, 8, n_samples)
        ).clip(0, 100)

        # Memory: gradual increase with occasional drops (restarts)
        base_mem = (
            50 + 20 * np.sin(2 * np.pi * hours / 24)
            + 0.01 * np.arange(n_samples)
            + np.random.normal(0, 5, n_samples)
        ).clip(20, 100)
        base_mem[base_mem > 90] = 45  # simulate restarts

        # Energy correlated with CPU
        energy = (base_cpu * 0.8 + base_mem * 0.3 + np.random.normal(0, 5, n_samples)).clip(10, 120)

        # Network I/O
        network_io = (
            20 + 30 * (hours > 8) * (hours < 18)
            + 10 * np.sin(2 * np.pi * hours / 24)
            + np.random.normal(0, 7, n_samples)
        ).clip(0, 100)

        # Disk I/O
        disk_io = (
            15 + 25 * np.random.exponential(1, n_samples)
            + 10 * (base_cpu > 70)
        ).clip(0, 100)

        # Inject anomalies
        anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        base_cpu[anomaly_idx]  = np.random.uniform(85, 100, len(anomaly_idx))
        base_mem[anomaly_idx]  = np.random.uniform(88, 100, len(anomaly_idx))
        energy[anomaly_idx]    = np.random.uniform(100, 120, len(anomaly_idx))

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30min'),
            'hour':      hours,
            'day':       days % 7,
            'cpu_usage':    np.round(base_cpu, 2),
            'memory_usage': np.round(base_mem, 2),
            'energy_consumption': np.round(energy, 2),
            'network_io': np.round(network_io, 2),
            'disk_io':    np.round(disk_io, 2),
        })

        # Ground-truth label
        def label_row(row):
            if row['cpu_usage'] > 85 and row['memory_usage'] > 85:
                return 'Critical'
            elif row['cpu_usage'] > 70 or row['memory_usage'] > 75:
                return 'High'
            elif row['cpu_usage'] < 20 and row['memory_usage'] < 40:
                return 'Idle'
            else:
                return 'Normal'

        df['behavior_label'] = df.apply(label_row, axis=1)
        return df


class ABISEngine:
    """Core ML engine: clustering, classification, anomaly detection"""

    def __init__(self):
        self.scaler       = StandardScaler()
        self.kmeans       = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.classifier   = RandomForestClassifier(n_estimators=100, random_state=42)
        self.iso_forest   = IsolationForest(contamination=0.05, random_state=42)
        self.pca          = PCA(n_components=2)
        self.feature_cols = ['cpu_usage', 'memory_usage', 'energy_consumption',
                             'network_io', 'disk_io', 'hour']
        self.is_trained   = False

    # ------------------------------------------------------------------ #
    def preprocess(self, df: pd.DataFrame):
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled

    # ------------------------------------------------------------------ #
    def train(self, df: pd.DataFrame):
        X, X_scaled = self.preprocess(df)

        # Clustering
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, cluster_labels)

        # Classification
        label_map = {'Idle': 0, 'Normal': 1, 'High': 2, 'Critical': 3}
        y = df['behavior_label'].map(label_map)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_tr, y_tr)
        acc = self.classifier.score(X_te, y_te)
        preds = self.classifier.predict(X_te)
        report = classification_report(y_te, preds,
                                       target_names=['Idle','Normal','High','Critical'],
                                       output_dict=True)

        # Anomaly detection
        anomaly_preds = self.iso_forest.fit_predict(X_scaled)

        # PCA for visualisation
        pca_coords = self.pca.fit_transform(X_scaled)

        self.is_trained = True
        return {
            'cluster_labels': cluster_labels.tolist(),
            'silhouette_score': round(sil, 4),
            'classification_accuracy': round(acc * 100, 2),
            'classification_report': report,
            'anomaly_flags': (anomaly_preds == -1).tolist(),
            'anomaly_count': int((anomaly_preds == -1).sum()),
            'pca_x': pca_coords[:, 0].tolist(),
            'pca_y': pca_coords[:, 1].tolist(),
            'feature_importance': dict(zip(
                self.feature_cols,
                [round(v, 4) for v in self.classifier.feature_importances_]
            )),
        }

    # ------------------------------------------------------------------ #
    def get_summary_stats(self, df: pd.DataFrame):
        return {
            'total_records': len(df),
            'avg_cpu': round(df['cpu_usage'].mean(), 2),
            'avg_memory': round(df['memory_usage'].mean(), 2),
            'avg_energy': round(df['energy_consumption'].mean(), 2),
            'peak_cpu': round(df['cpu_usage'].max(), 2),
            'peak_memory': round(df['memory_usage'].max(), 2),
            'label_distribution': df['behavior_label'].value_counts().to_dict(),
            'hourly_avg': df.groupby('hour')[['cpu_usage','memory_usage','energy_consumption']].mean().round(2).to_dict(),
        }

    # ------------------------------------------------------------------ #
    def get_optimization_recommendations(self, df: pd.DataFrame):
        recs = []
        avg_cpu    = df['cpu_usage'].mean()
        avg_mem    = df['memory_usage'].mean()
        avg_energy = df['energy_consumption'].mean()
        crit_pct   = (df['behavior_label'] == 'Critical').mean() * 100
        idle_pct   = (df['behavior_label'] == 'Idle').mean() * 100

        if avg_cpu > 70:
            recs.append({'level': 'critical', 'title': 'CPU Overload Detected',
                         'desc': f'Average CPU at {avg_cpu:.1f}%. Scale horizontally or optimize compute-heavy processes.',
                         'impact': 'High', 'effort': 'Medium'})
        if avg_mem > 75:
            recs.append({'level': 'warning', 'title': 'Memory Pressure',
                         'desc': f'Average memory at {avg_mem:.1f}%. Investigate memory leaks and increase swap allocation.',
                         'impact': 'High', 'effort': 'Low'})
        if avg_energy > 80:
            recs.append({'level': 'warning', 'title': 'High Energy Consumption',
                         'desc': f'Average energy at {avg_energy:.1f} units. Enable CPU frequency scaling and consolidate idle VMs.',
                         'impact': 'Medium', 'effort': 'Low'})
        if crit_pct > 5:
            recs.append({'level': 'critical', 'title': f'Critical States: {crit_pct:.1f}% of time',
                         'desc': 'System spends excessive time in critical state. Implement auto-scaling policies.',
                         'impact': 'Critical', 'effort': 'Medium'})
        if idle_pct > 30:
            recs.append({'level': 'info', 'title': f'Idle Time: {idle_pct:.1f}%',
                         'desc': 'High idle periods indicate resource over-provisioning. Right-size instances.',
                         'impact': 'Medium', 'effort': 'Low'})
        if not recs:
            recs.append({'level': 'success', 'title': 'System Healthy',
                         'desc': 'No critical inefficiencies detected. Continue monitoring.',
                         'impact': 'None', 'effort': 'None'})
        return recs
