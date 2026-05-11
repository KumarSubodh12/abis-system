"""
ABIS Web Server - Flask Backend
Adaptive Behavioral Intelligence System
"""

from flask import Flask, jsonify, render_template, request
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from models.abis_engine import DataGenerator, ABISEngine

app = Flask(__name__)

# Global state
engine = ABISEngine()
df     = None
results = {}


def _ensure_trained():
    global df, results
    if df is None:
        df = DataGenerator.generate(n_samples=2000)
        results = engine.train(df)
    return df, results


# ------------------------------------------------------------------ #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def api_train():
    global df, results, engine
    n = request.json.get('n_samples', 2000)
    engine = ABISEngine()
    df = DataGenerator.generate(n_samples=n)
    results = engine.train(df)
    return jsonify({'status': 'ok', 'message': f'Trained on {n} samples'})


@app.route('/api/summary')
def api_summary():
    d, _ = _ensure_trained()
    return jsonify(engine.get_summary_stats(d))


@app.route('/api/results')
def api_results():
    _ensure_trained()
    return jsonify({
        'silhouette_score':          results['silhouette_score'],
        'classification_accuracy':   results['classification_accuracy'],
        'anomaly_count':             results['anomaly_count'],
        'feature_importance':        results['feature_importance'],
        'classification_report':     results['classification_report'],
    })


@app.route('/api/timeseries')
def api_timeseries():
    d, _ = _ensure_trained()
    sample = d.iloc[::4].copy()  # every 4th row for perf
    return jsonify({
        'timestamps':   sample['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'cpu':          sample['cpu_usage'].tolist(),
        'memory':       sample['memory_usage'].tolist(),
        'energy':       sample['energy_consumption'].tolist(),
        'network':      sample['network_io'].tolist(),
        'disk':         sample['disk_io'].tolist(),
        'labels':       sample['behavior_label'].tolist(),
        'anomalies':    [results['anomaly_flags'][i] for i in range(0, len(d), 4)],
    })


@app.route('/api/clusters')
def api_clusters():
    d, r = _ensure_trained()
    step = 10
    return jsonify({
        'pca_x':   r['pca_x'][::step],
        'pca_y':   r['pca_y'][::step],
        'clusters': r['cluster_labels'][::step],
        'labels':  d['behavior_label'].tolist()[::step],
        'cpu':     d['cpu_usage'].tolist()[::step],
        'memory':  d['memory_usage'].tolist()[::step],
    })


@app.route('/api/hourly')
def api_hourly():
    d, _ = _ensure_trained()
    h = d.groupby('hour')[['cpu_usage','memory_usage','energy_consumption','network_io']].mean().round(2)
    return jsonify({
        'hours':   h.index.tolist(),
        'cpu':     h['cpu_usage'].tolist(),
        'memory':  h['memory_usage'].tolist(),
        'energy':  h['energy_consumption'].tolist(),
        'network': h['network_io'].tolist(),
    })


@app.route('/api/recommendations')
def api_recommendations():
    d, _ = _ensure_trained()
    return jsonify(engine.get_optimization_recommendations(d))


@app.route('/api/label_distribution')
def api_label_dist():
    d, r = _ensure_trained()
    dist = d['behavior_label'].value_counts().to_dict()
    return jsonify(dist)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ABIS — Adaptive Behavioral Intelligence System")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("="*60 + "\n")
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)