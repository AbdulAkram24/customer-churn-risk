import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / "data" / "dataset_01_customer_churn_risk.xlsx"
df = pd.read_excel(dataset_path, sheet_name="Sheet1")
expected_columns = [
    "tenure_months", "monthly_charges", "support_tickets",
    "avg_session_minutes", "late_payments", "contract_months", "target"
]
if df.columns.tolist() != expected_columns:
    raise ValueError(
        f"Expected Excel headings {expected_columns}, found {df.columns.tolist()}"
    )

print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Class balance:\n", df['target'].value_counts(normalize=True))

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))

coefficients = pd.DataFrame({
    "Feature": df.drop("target", axis=1).columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients):")
print(coefficients)

for feature, coef in zip(df.drop("target", axis=1).columns, log_reg.coef_[0]):
    direction = "increases" if coef > 0 else "decreases"
    print(f"- Higher {feature} {direction} churn risk (coef={coef:.3f})")
