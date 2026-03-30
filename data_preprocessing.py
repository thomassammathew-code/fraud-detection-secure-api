import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os


if not os.path.exists("model"):
    os.makedirs("model")
df = pd.read_csv("data/creditcard.csv")
df = df.sample(n=50000, random_state=42)
print("Dataset Loaded (Sampled)\n")

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(["Time"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]
print("Preprocessing Done")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\nAfter SMOTE:")
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)
print("\nTrain/Test Split Done")
model = RandomForestClassifier(
    n_estimators=50,   
    max_depth=10,      
    random_state=42,
    n_jobs=-1          
)
model.fit(X_train, y_train)
print("\nModel Training Completed")
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "model/fraud_model.pkl")
print("\nModel Saved Successfully")
