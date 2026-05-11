# ⬡ ABIS — Adaptive Behavioral Intelligence System
### Resource Utilization Optimization Engine

MADE BY KUMAR SUBODH


---

## 📁 Project Structure

```
ABIS/
├── app.py                  ← Flask web server (run this for dashboard)
├── analyze.py              ← Standalone CLI analysis + chart export
├── requirements.txt        ← Python dependencies
├── models/
│   ├── __init__.py
│   └── abis_engine.py      ← Core AI/ML engine
└── templates/
    └── index.html          ← Premium web dashboard UI
```

---

## 🚀 Quick Start (VS Code)

### Step 1 — Open in VS Code
```
File → Open Folder → select the ABIS/ folder
```

### Step 2 — Create virtual environment (in VS Code terminal)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4A — Run the Web Dashboard
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

### Step 4B — Run CLI Analysis (no browser needed)
```bash
python analyze.py
```
This will print all results and save `abis_analysis.png`.

---

## 🤖 ML Pipeline

| Model | Purpose | Algorithm |
|---|---|---|
| **K-Means Clustering** | Group usage behaviors | Unsupervised (k=4) |
| **Random Forest** | Classify resource states | Ensemble (100 trees) |
| **Isolation Forest** | Detect anomalies | Outlier detection |
| **PCA** | Visualize clusters | Dimensionality reduction |

### Behavioral Classes
- **Idle** — CPU < 20%, Memory < 40%
- **Normal** — Typical operating range
- **High** — CPU > 70% or Memory > 75%
- **Critical** — CPU > 85% AND Memory > 85%

---

## 📊 Features Analyzed
- CPU Usage (%)
- Memory Usage (%)
- Energy Consumption (units)
- Network I/O (%)
- Disk I/O (%)
- Hour of day (temporal pattern)

---

## 🌐 Web Dashboard Sections
1. **Hero** — System overview with live stats
2. **Overview** — 6 metric cards with animated bars
3. **Analytics** — 5 interactive Chart.js visualizations
4. **AI Results** — Accuracy rings, feature importance, anomaly count
5. **Train Panel** — Retrain ML model live in browser
6. **Recommendations** — AI-generated optimization strategies

---

## 🛠 Technology Stack
- **Python** — Core language
- **NumPy / Pandas** — Data processing
- **Scikit-learn** — ML models
- **Matplotlib** — Static visualization
- **Flask** — Web server
- **Chart.js** — Interactive charts
- **Three.js** — 3D particle background

---

*ABIS v2.4 — Adaptive Behavioral Intelligence System*
