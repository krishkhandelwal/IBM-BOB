# 📡 Telco Customer Churn Prediction

A full-stack machine learning application built with **Python**, **scikit-learn**, and **Streamlit** that predicts which telecom customers are likely to churn — empowering proactive retention decisions.

---

- **Backend**: Python (scikit-learn, pandas, joblib)
- **Frontend**: Streamlit
- **ML Model**: Logistic Regression, Random Forest, Gradient Boosting (scikit-learn)
- **Dataset**: `data/Telco_customer_churn.xlsx` (7 043 customers, 33 features)

---

## 🗂 Project Structure

```
churn_prediction/
├── app.py                    # Main Streamlit entry point
├── requirements.txt
├── data/
│   └── Telco_customer_churn.xlsx
├── models/                   # Trained model artefacts (.pkl)
├── utils/
│   ├── preprocessing.py      # Data loading & feature engineering
│   ├── model.py              # Training, evaluation & persistence
│   └── session.py            # Streamlit session state management
└── pages/
    ├── home.py               # Landing page & project overview
    ├── eda.py                # Exploratory Data Analysis
    ├── model_performance.py  # ROC, confusion matrix, feature importances
    ├── predict.py            # Interactive single-customer prediction
    └── dataset_viewer.py     # Filter & download raw data
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
cd churn_prediction
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline linear model with StandardScaler |
| **Random Forest** | 200 trees, max depth 12 — best interpretability |
| **Gradient Boosting** | 150 estimators, learning rate 0.1 — typically best AUC |

The best model (by ROC-AUC) is auto-selected and saved as `models/best_model.pkl`.

---

## 📊 Dataset

- **Source**: IBM Telco Customer Churn (7 043 customers, 33 features)
- **Target**: `Churn Value` (1 = churned, 0 = retained)
- **Churn rate**: ~26.5%

### Feature Engineering
- Dropped identifiers, geographic coordinates, and data-leakage columns (`CLTV`, `Churn Score`, `Churn Reason`)
- Binary encoded Yes/No columns
- One-hot encoded multi-class categoricals (Contract, Payment Method, Internet Service, etc.)
- Imputed missing `Total Charges` with `Monthly Charges`

---

## 🖥 Application Pages

| Page | Description |
|---|---|
| **🏠 Home** | KPI dashboard & project overview |
| **📊 Exploratory Analysis** | Distribution charts, categorical breakdowns, churn reasons, correlation heatmap |
| **🤖 Model Performance** | Metrics table, ROC curves, confusion matrices, feature importances |
| **🔮 Predict Churn** | Interactive form → churn probability + risk badge + gauge chart |
| **📋 Dataset Viewer** | Multi-filter table with CSV download |

---

## 📈 Typical Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.81 | ~0.67 | ~0.55 | ~0.60 | ~0.85 |
| Random Forest | ~0.80 | ~0.65 | ~0.52 | ~0.58 | ~0.84 |
| Gradient Boosting | ~0.82 | ~0.69 | ~0.57 | ~0.62 | **~0.86** |

*(Exact numbers vary by random seed and train/test split)*
