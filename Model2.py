import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel("Dataset_2000.xlsx", engine="openpyxl")

print("Original Shape:", df.shape)

# ===============================
# TARGET
# ===============================
TARGET = "PCOS (Y/N)"
y = df[TARGET].astype(int)

# ===============================
# DROP LEAKAGE / HIGH-RISK FEATURES
# ===============================
DROP_COLS = [
    TARGET,
    "AMH(ng/mL)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (R) (mm)",
    "Avg. F size (L) (mm)",
    "FSH/LH",
    "Sl. No",
    "Patient File No.",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "Blood Group",
    "Ibeta-HCG(mIU/mL)",
    "IIbeta-HCG(mIU/mL)",
    "Pregnant"
]

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# NUMERIC CLEANING (FIX beta-HCG BUG)
# ===============================
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X.fillna(X.median(), inplace=True)
print("Any NaNs left?", X.isna().sum().sum())

# ===============================
# ONE-HOT ENCODING
# ===============================
X = pd.get_dummies(X, drop_first=True)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# SCALER (KEPT FOR STRUCTURE)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

log_reg.fit(X_train_scaled, y_train)

y_lr_pred = log_reg.predict(X_test_scaled)
y_lr_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression")
print("---------------------------------------------")
print("Accuracy :", accuracy_score(y_test, y_lr_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_lr_prob))
print(classification_report(y_test, y_lr_pred))


# ===============================
# MODEL (REGULARIZED → LESS OVERFITTING)
# ===============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nXGBoost")
print("---------------------------------------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# ===============================
# SAVE ARTIFACTS
# ===============================
joblib.dump(model, "pcos_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")

# ===============================
# SHAP (KEPT)
# ===============================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(
    shap_values,
    X_train,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# ===============================
# CLEAN STREAMLIT TEST CSV (NO TARGET)
# ===============================
X_test.sample(1).to_csv(
    "streamlit_test_input.csv",
    index=False
)


print("\nModel, scaler, schema & SHAP saved successfully.")
