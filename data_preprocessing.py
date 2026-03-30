import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# ================================
# STEP 0: Ensure model folder exists
# ================================
if not os.path.exists("model"):
    os.makedirs("model")

# ================================
# STEP 1: Load Dataset (USE SAMPLE FOR SPEED)
# ================================
df = pd.read_csv("data/creditcard.csv")

# 🔥 TAKE ONLY SMALL SAMPLE (FASTER)
df = df.sample(n=50000, random_state=42)

print("Dataset Loaded (Sampled)\n")

# ================================
# STEP 2: Preprocessing
# ================================
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

df = df.drop(["Time"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

print("Preprocessing Done")

# ================================
# STEP 3: Handle Imbalance (SMOTE)
# ================================
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(y_resampled.value_counts())

# ================================
# STEP 4: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

print("\nTrain/Test Split Done")

# ================================
# STEP 5: Train Model (FASTER SETTINGS)
# ================================
model = RandomForestClassifier(
    n_estimators=50,   # reduced trees
    max_depth=10,      # limit depth
    random_state=42,
    n_jobs=-1          # use all CPU cores
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ================================
# STEP 6: Evaluation
# ================================
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# STEP 7: Save Model
# ================================
joblib.dump(model, "model/fraud_model.pkl")

print("\nModel Saved Successfully")