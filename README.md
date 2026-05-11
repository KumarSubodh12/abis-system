# ⬡ ABIS — Adaptive Behavioral Intelligence System

### Resource Utilization Optimization Engine  
**Made by Kumar Subodh**  

🔗 **Deployment Link:**  
https://abis-system.onrender.com/

---

# 📁 Project Structure

```bash
ABIS/
├── app.py                  # Flask web server (run this for dashboard)
├── analyze.py              # Standalone CLI analysis + chart export
├── requirements.txt        # Python dependencies
├── models/
│   ├── __init__.py
│   └── abis_engine.py      # Core AI/ML engine
└── templates/
    └── index.html          # Premium web dashboard UI
```

---

# 🚀 Quick Start (VS Code)

## Step 1 — Open in VS Code

Go to:

```bash
File → Open Folder → Select the ABIS/ folder
```

---

## Step 2 — Create Virtual Environment

Open VS Code terminal and run:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4A — Run the Web Dashboard

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

in your browser.

---

## Step 4B — Run CLI Analysis (No Browser Needed)

```bash
python analyze.py
```

This will:

- Print complete analysis results
- Save visualization as:

```bash
abis_analysis.png
```

---

# 🤖 ML Pipeline

| Model | Purpose | Algorithm |
|------|---------|-----------|
| **K-Means Clustering** | Group usage behaviors | Unsupervised (`k=4`) |
| **Random Forest** | Classify resource states | Ensemble (`100 trees`) |
| **Isolation Forest** | Detect anomalies | Outlier detection |
| **PCA** | Visualize clusters | Dimensionality reduction |

---

# 🧠 Behavioral Classes

### **Idle**
- CPU < 20%
- Memory < 40%

### **Normal**
- Typical operating range

### **High**
- CPU > 70%
- Memory > 75%

### **Critical**
- CPU > 85%
- Memory > 85%

---

# 📊 Features Analyzed

- CPU Usage (%)
- Memory Usage (%)
- Energy Consumption (units)
- Network I/O (%)
- Disk I/O (%)
- Hour of Day (temporal pattern)

---

# 🌐 Web Dashboard Sections

### Hero
System overview with live stats

### Overview
6 metric cards with animated bars

### Analytics
5 interactive **Chart.js** visualizations

### AI Results
- Accuracy rings  
- Feature importance  
- Anomaly count

### Train Panel
Retrain ML model live in browser

### Recommendations
AI-generated optimization strategies

---

# 🛠 Technology Stack

- **Python** — Core language
- **NumPy / Pandas** — Data processing
- **Scikit-learn** — ML models
- **Matplotlib** — Static visualization
- **Flask** — Web server
- **Chart.js** — Interactive charts
- **Three.js** — 3D particle background

---

# ⚡ Version

**ABIS v2.4 — Adaptive Behavioral Intelligence System**
