# ===========================================
# 🛡️ Fraud Detection Model Training Script
# ===========================================

# 📦 Install necessary packages (for Colab users)
!pip install pandas scikit-learn xgboost joblib seaborn matplotlib

# 📚 Import libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# 1️⃣ Load Dataset
# ===========================================

# Load the dataset (download manually from Kaggle and upload to Colab or local)
df = pd.read_csv('fraud_dataset.csv')

# Preview
df.head()

# ===========================================
# 2️⃣ Preprocess Data
# ===========================================

# Use only selected predictors
predictors = ['transaction_amount', 'Location', 'Age', 'Gender']
target = 'fraud_label'

# Drop NA
df = df[predictors + [target]].dropna()

# Encode categorical columns
label_encoders = {}
for col in ['Location', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and labels
X = df[predictors]
y = df[target]

# ===========================================
# 3️⃣ Train-Test-Validation Split (50-30-20)
# ===========================================

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.375, stratify=y_temp, random_state=42)  # 0.375*0.8 ≈ 0.3

# ===========================================
# 4️⃣ Train Multiple Models and Evaluate
# ===========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_model_name = None
best_auc = 0

print("🔍 Model Evaluation:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print(f"{name} AUC: {auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name} (AUC = {best_auc:.4f})")

# ===========================================
# 5️⃣ Final Evaluation on Test Set
# ===========================================

y_pred_test = best_model.predict(X_test)
print("\n📊 Classification Report (Test Set):\n")
print(classification_report(y_test, y_pred_test))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.show()

# ===========================================
# 6️⃣ Save Model and Encoders
# ===========================================

joblib.dump(best_model, "best_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n💾 Model and encoders saved as 'best_model.pkl' and 'label_encoders.pkl'")
