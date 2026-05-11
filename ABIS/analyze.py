"""
ABIS Standalone Analyzer — run without Flask
Generates charts and prints results to console
Usage: python analyze.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from models.abis_engine import DataGenerator, ABISEngine

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Color palette ──────────────────────────────────────────────────────────
DARK   = '#040812'
SURF   = '#0c1a2e'
SURF2  = '#112240'
ACCENT = '#00d4ff'
GREEN  = '#06ffa5'
PURPLE = '#7c3aed'
WARN   = '#f59e0b'
DANGER = '#ef4444'
TEXT   = '#e2e8f0'
MUTED  = '#64748b'

plt.rcParams.update({
    'figure.facecolor': DARK,
    'axes.facecolor':   SURF,
    'axes.edgecolor':   SURF2,
    'axes.labelcolor':  TEXT,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT,
    'grid.color':       SURF2,
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
    'font.size':        9,
})

CLUSTER_COLORS = [ACCENT, GREEN, WARN, DANGER]
LABEL_MAP = {0:'Idle', 1:'Normal', 2:'High', 3:'Critical'}


def banner():
    print("\n" + "═"*62)
    print("  ⬡  ABIS — Adaptive Behavioral Intelligence System")
    print("     Resource Utilization Optimization Engine")
    print("═"*62)


def run():
    banner()

    # 1. Generate data
    print("\n[1/5] Generating synthetic dataset (2000 samples)...")
    df = DataGenerator.generate(n_samples=2000)
    print(f"      Rows: {len(df)} | Cols: {list(df.columns)}")

    # 2. Train
    print("[2/5] Training ML models...")
    engine = ABISEngine()
    results = engine.train(df)
    stats   = engine.get_summary_stats(df)
    recs    = engine.get_optimization_recommendations(df)

    # 3. Print results
    print("\n[3/5] Model Results")
    print(f"      Classification Accuracy : {results['classification_accuracy']}%")
    print(f"      Silhouette Score        : {results['silhouette_score']}")
    print(f"      Anomalies Detected      : {results['anomaly_count']}")
    print(f"      Avg CPU                 : {stats['avg_cpu']}%")
    print(f"      Avg Memory              : {stats['avg_memory']}%")
    print(f"      Avg Energy              : {stats['avg_energy']} units")
    print(f"\n      Feature Importance:")
    for f, v in sorted(results['feature_importance'].items(), key=lambda x:-x[1]):
        bar = '█' * int(v * 50)
        print(f"        {f:<22} {bar:<30} {v:.4f}")

    print("\n[4/5] Generating visualizations...")

    # 4. Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=DARK)
    fig.suptitle('ABIS — Resource Behavioral Analysis Dashboard', fontsize=16,
                 color=TEXT, fontweight='bold', y=.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=.45, wspace=.35,
                           top=.93, bottom=.06, left=.06, right=.97)

    # ── 1. Timeline (full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    sample = df.iloc[::2]
    ax1.plot(sample.index, sample['cpu_usage'],    color=ACCENT,  lw=1.2, label='CPU %',    alpha=.9)
    ax1.plot(sample.index, sample['memory_usage'], color=PURPLE,  lw=1.2, label='Memory %', alpha=.9)
    ax1.plot(sample.index, sample['energy_consumption'], color=GREEN, lw=1.0, label='Energy', alpha=.7)
    anom_idx = [i for i,v in enumerate(results['anomaly_flags']) if v]
    if anom_idx:
        ax1.scatter(anom_idx, df.iloc[anom_idx].cpu_usage,
                   color=DANGER, s=12, zorder=5, label='Anomaly', alpha=.9)
    ax1.set_title('System Resource Timeline with Anomaly Markers', color=TEXT, pad=8)
    ax1.legend(loc='upper right', framealpha=.2, labelcolor=TEXT)
    ax1.grid(True, alpha=.3)

    # ── 2. Hourly heatmap ──
    ax2 = fig.add_subplot(gs[1, 0])
    hourly = df.groupby('hour')[['cpu_usage','memory_usage','energy_consumption','network_io','disk_io']].mean()
    cmap = LinearSegmentedColormap.from_list('abis', [SURF, ACCENT, GREEN])
    im = ax2.imshow(hourly.T, aspect='auto', cmap=cmap, interpolation='bilinear')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['CPU','Mem','Energy','Net','Disk'], fontsize=8)
    ax2.set_xlabel('Hour of Day')
    ax2.set_title('Hourly Resource Heatmap', color=TEXT, pad=8)
    plt.colorbar(im, ax=ax2, fraction=.04, pad=.04)

    # ── 3. PCA Scatter ──
    ax3 = fig.add_subplot(gs[1, 1])
    pca_x = np.array(results['pca_x'])
    pca_y = np.array(results['pca_y'])
    clusters = np.array(results['cluster_labels'])
    for ci, col in enumerate(CLUSTER_COLORS):
        mask = clusters == ci
        ax3.scatter(pca_x[mask], pca_y[mask], c=col, s=6, alpha=.5, label=f'C{ci}')
    ax3.set_title('PCA Cluster Space (K-Means)', color=TEXT, pad=8)
    ax3.legend(loc='upper right', framealpha=.2, labelcolor=TEXT, markerscale=2)
    ax3.grid(True, alpha=.3)

    # ── 4. Label distribution ──
    ax4 = fig.add_subplot(gs[1, 2])
    dist = stats['label_distribution']
    labels = list(dist.keys())
    vals   = list(dist.values())
    bar_colors = [GREEN if l=='Normal' else WARN if l=='High' else DANGER if l=='Critical' else MUTED for l in labels]
    bars = ax4.bar(labels, vals, color=bar_colors, edgecolor=SURF, linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(val),
                ha='center', color=TEXT, fontsize=8)
    ax4.set_title('Behavioral State Distribution', color=TEXT, pad=8)
    ax4.grid(True, axis='y', alpha=.3)

    # ── 5. Feature importance ──
    ax5 = fig.add_subplot(gs[2, 0])
    fi = results['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1])
    fi_labels = [x[0].replace('_',' ') for x in fi_sorted]
    fi_vals   = [x[1] for x in fi_sorted]
    cmap_fi = LinearSegmentedColormap.from_list('fi', [PURPLE, ACCENT])
    fi_colors = [cmap_fi(v/max(fi_vals)) for v in fi_vals]
    ax5.barh(fi_labels, fi_vals, color=fi_colors, edgecolor=SURF, height=.7)
    ax5.set_title('Feature Importance (Random Forest)', color=TEXT, pad=8)
    ax5.grid(True, axis='x', alpha=.3)

    # ── 6. Hourly line chart ──
    ax6 = fig.add_subplot(gs[2, 1])
    h = df.groupby('hour')[['cpu_usage','memory_usage','energy_consumption']].mean()
    ax6.plot(h.index, h['cpu_usage'],    color=ACCENT,  lw=2, label='CPU')
    ax6.plot(h.index, h['memory_usage'], color=PURPLE,  lw=2, label='Memory')
    ax6.plot(h.index, h['energy_consumption'], color=GREEN, lw=2, label='Energy')
    ax6.fill_between(h.index, h['cpu_usage'], alpha=.1, color=ACCENT)
    ax6.set_title('24-Hour Average Profile', color=TEXT, pad=8)
    ax6.legend(loc='upper left', framealpha=.2, labelcolor=TEXT)
    ax6.set_xlabel('Hour')
    ax6.grid(True, alpha=.3)

    # ── 7. Classification report heatmap ──
    ax7 = fig.add_subplot(gs[2, 2])
    report = results['classification_report']
    classes = ['Idle','Normal','High','Critical']
    metrics = ['precision','recall','f1-score']
    mat = np.array([[report.get(c,{}).get(m,0) for m in metrics] for c in classes])
    cmap_rep = LinearSegmentedColormap.from_list('rep', [SURF2, PURPLE, ACCENT])
    im2 = ax7.imshow(mat, cmap=cmap_rep, vmin=0, vmax=1, aspect='auto')
    ax7.set_xticks(range(3)); ax7.set_xticklabels(['Precision','Recall','F1'], fontsize=8)
    ax7.set_yticks(range(4)); ax7.set_yticklabels(classes, fontsize=8)
    for i in range(4):
        for j in range(3):
            ax7.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=9, color=TEXT)
    ax7.set_title('Classification Report', color=TEXT, pad=8)
    plt.colorbar(im2, ax=ax7, fraction=.05, pad=.04)

    out_path = os.path.join(os.path.dirname(__file__), 'abis_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f"      Saved → {out_path}")

    # 5. Recommendations
    print("\n[5/5] Optimization Recommendations")
    icons = {'critical':'🚨','warning':'⚠️','info':'💡','success':'✅'}
    for r in recs:
        print(f"\n  {icons.get(r['level'],'•')} [{r['level'].upper()}] {r['title']}")
        print(f"     {r['desc']}")
        print(f"     Impact: {r['impact']}  |  Effort: {r['effort']}")

    print("\n" + "═"*62)
    print("  Analysis complete. Open abis_analysis.png to view charts.")
    print("  Run app.py to launch the interactive web dashboard.")
    print("═"*62 + "\n")
    plt.show()


if __name__ == '__main__':
    run()
