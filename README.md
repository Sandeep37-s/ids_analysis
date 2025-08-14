.
├─ data/
│  ├─ KDDTrain+.txt
│  ├─ KDDTest+.txt
│  └─ README_DATA.md        # optional notes about dataset source
├─ src/
│  ├─ preprocess.py         # load + clean + encode + scale
│  ├─ train.py              # trains models & saves artifacts
│  ├─ evaluate.py           # metrics, confusion matrix, ROC
│  ├─ utils.py              # shared helpers (logging, seeds, I/O)
│  └─ config.py             # central config (paths, params)
├─ notebooks/
│  └─ EDA.ipynb             # exploratory data analysis
├─ models/                  # saved models (*.joblib) & encoders/scalers
├─ reports/
│  ├─ metrics.csv           # per-model metrics
│  ├─ confusion_matrix_*.png
│  └─ roc_*.png
├─ requirements.txt
└─ README.md


🔍 Dataset (NSL-KDD)

Files used: KDDTrain+.txt (train) and KDDTest+.txt (test).

Features: 41 network traffic features + 1 label column (attack name or normal).

Attack families (multi-class option):

DoS (e.g., smurf, neptune)

Probe (e.g., satan, portsweep)

R2L (e.g., guess_passwd, ftp_write)

U2R (e.g., buffer_overflow, rootkit)

Tasks supported:

Binary: normal vs attack

Multi-class: normal, DoS, Probe, R2L, U2R

Place the two NSL-KDD files under data/ with the exact names shown above.

🛠️ Setup
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt


requirements.txt (example)
Adjust as needed:

pandas
numpy
scikit-learn
matplotlib
joblib
tqdm

⚙️ Configuration

Edit defaults in src/config.py:

DATA_DIR = "data"
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"

TASK = "binary"  # "binary" or "multiclass"
RANDOM_STATE = 42
TEST_SIZE = 0.0  # use external test set by default

# Preprocessing
SCALE_NUMERIC = True
IMPUTE_NUMERIC = True
IMPUTE_CATEGORICAL = True
HANDLE_IMBALANCE = "class_weight"  # or "none" / "smote" (if you add imblearn)

# Models to run
MODELS = {
    "logreg": {"type": "LogisticRegression", "params": {"max_iter": 200, "n_jobs": None}},
    "rf":     {"type": "RandomForestClassifier", "params": {"n_estimators": 200, "n_jobs": -1}},
    "xgb":    {"type": "XGBClassifier", "params": {"n_estimators": 300, "n_jobs": -1}},  # if you add xgboost
    "svm":    {"type": "SVC", "params": {"kernel": "rbf", "probability": True}},
    "knn":    {"type": "KNeighborsClassifier", "params": {"n_neighbors": 7}},
    "nb":     {"type": "GaussianNB", "params": {}},
}


If you don’t plan to use XGBoost, remove it from MODELS and from requirements.txt.

🧹 Preprocessing Pipeline

Load KDDTrain+.txt and KDDTest+.txt with known column names.

Map raw label strings → binary (normal→0, attack→1) or → 5-class (normal, DoS, Probe, R2L, U2R).

Split features:

Categorical: protocol_type, service, flag

Numeric: remaining 38 features

Handle missing (rare) values (impute).

Encode categoricals with One-Hot Encoding.

Scale numeric features with StandardScaler (optional).

Save fitted encoders/scalers to models/ for consistent evaluation.

🚀 Run
# 1) Preprocess and save transformed arrays + artifacts
python -m src.preprocess

# 2) Train all configured models (saves .joblib + per-model reports)
python -m src.train

# 3) Evaluate on held-out NSL-KDD test set (saves metrics & plots)
python -m src.evaluate


Artifacts and figures are written under models/ and reports/.

📊 Metrics & Reporting

For binary and multi-class tasks:

Overall: Accuracy, Precision, Recall, F1-score, ROC-AUC (macro/weighted for multi-class)

Per-class report (sklearn classification_report)

Confusion matrix (reports/confusion_matrix_<model>.png)

ROC curves (reports/roc_<model>.png)

Example results table (replace with your runs):

Model	Task	Accuracy	Precision	Recall	F1	ROC-AUC
LogReg	binary	0.93	0.93	0.93	0.93	0.97
RandomForest	binary	0.96	0.96	0.96	0.96	0.99
SVM (RBF)	binary	0.95	0.95	0.95	0.95	0.98
KNN	binary	0.92	0.92	0.92	0.92	0.96
NaiveBayes	binary	0.86	0.86	0.86	0.86	0.90

Note: NSL-KDD is class-imbalanced in family labels—prefer macro/weighted scores and show per-class metrics.

🧪 Reproducibility

Set global random seeds (RANDOM_STATE) across numpy/sklearn.

Persist fitted preprocessors (encoder/scaler) and models with joblib.

Record params & metrics to reports/metrics.csv.

🧭 Command Line Options (optional)

You can add simple CLI flags in train.py/evaluate.py, e.g.:

python -m src.train --task multiclass --models rf,svm,xgb
python -m src.evaluate --task binary --model rf

🧩 Using the Trained Model in Your App
# example_infer.py
import joblib
import numpy as np

enc = joblib.load("models/onehot_encoder.joblib")
scaler = joblib.load("models/scaler.joblib")
clf = joblib.load("models/rf.joblib")

# X_raw: dict or DataFrame row with original 41 features
# 1) build DataFrame with correct columns
# 2) transform categorical via enc, numeric via scaler
# 3) concatenate → X_ready
y_pred = clf.predict(X_ready)


Keep the exact same preprocessing steps at inference as during training.

🧠 Tips

Try class_weight="balanced" for Logistic Regression, SVM, and RandomForest.

Consider SMOTE (requires imbalanced-learn) for multi-class when minority classes are tiny.

Hyperparameter tune with GridSearchCV/RandomizedSearchCV and stratified folds.

✅ Checklist

 Place KDDTrain+.txt / KDDTest+.txt under data/

 Run preprocessing

 Train models (binary &/or multi-class)

 Evaluate and export figures

 Update results table in this README

📚 Citation

If you use this project in academic work, please cite the original NSL-KDD dataset paper and any libraries you rely on.

📄 License

This project is released under the MIT License. See LICENSE for details.
