# src/train_model.py
import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ✅ Path to your CSV file
csv_path = r"C:\Users\HP\OneDrive\Desktop\SBS HACKATHON\b2b-saas-usage-retention.csv"

print(f"📂 Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# --- Basic Cleaning ---
print(f"✅ Dataset loaded | Shape: {df.shape}")

target = "churn_risk_score"
if target not in df.columns:
    raise KeyError(f"❌ Target column '{target}' not found in dataset!")

# --- Convert target to binary churn flag ---
df["churn_label"] = (df[target] >= 0.5).astype(int)

# --- Handle date columns ---
date_cols = [c for c in df.columns if "date" in c.lower()]
for col in date_cols:
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        pass

# Create useful date-based numeric features (days since start, etc.)
if "subscription_start_date" in df.columns and "subscription_end_date" in df.columns:
    df["subscription_duration_days"] = (
        (df["subscription_end_date"] - df["subscription_start_date"]).dt.days
    )

if "last_login_date" in df.columns:
    df["days_since_last_login"] = (
        pd.Timestamp.now() - df["last_login_date"]
    ).dt.days

if "last_success_touch_date" in df.columns:
    df["days_since_last_touch"] = (
        pd.Timestamp.now() - df["last_success_touch_date"]
    ).dt.days

# --- Drop non-informative text fields ---
drop_cols = [
    "customer_name", "account_manager", "notes",
    "subscription_start_date", "subscription_end_date",
    "last_login_date", "last_success_touch_date",
    target, "customer_id"
]
feature_cols = [c for c in df.columns if c not in drop_cols and c != "churn_label"]

# --- Encode categorical columns ---
cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# --- Final Feature Matrix ---
X = df[feature_cols].fillna(0)
y = df["churn_label"]

print(f"📊 Final feature matrix: {X.shape[1]} features")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train LightGBM Model ---
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("🚀 Training LightGBM model...")
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"✅ Model trained successfully | AUC: {auc:.3f}")

# --- Save Model ---
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "lightgbm_churn_model.pkl")

joblib.dump(model, model_path)
print(f"💾 Model saved successfully to: {os.path.abspath(model_path)}")
