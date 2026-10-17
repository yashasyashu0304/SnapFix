import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import numpy as np

# Load data
df = pd.read_csv("complaints_text_dataset.csv")

X = df["text"]
y_true = df["label"].astype(str)   # force string labels

# Load model
vectorizer = joblib.load("text_vectorizer.joblib")
model = joblib.load("text_classifier.joblib")

# Predict
X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)

# Ensure predictions are strings
y_pred = y_pred.astype(str)

# Get labels safely
labels = sorted(y_true.unique())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nPer-class Accuracy (Text Model):\n")

for i, label in enumerate(labels):
    support = cm[i].sum()
    if support == 0:
        print(f"{label:30s} : N/A")
    else:
        acc = cm[i, i] / support
        print(f"{label:30s} : {acc:.4f}")