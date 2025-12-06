import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("=" * 50)
print("ML Model Training Started")
print("=" * 50)

# Load Training Data
print("\n[1/6] Loading Training.csv...")
train_data = pd.read_csv('datasets/Training.csv')
print(f"✓ Loaded {len(train_data)} records")

# Separate features and target
X = train_data.drop('prognosis', axis=1)
y = train_data['prognosis']

# Handle missing values (Fill NaN with 0)
print("\n[2/6] Cleaning data...")
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"⚠ Found {missing_count} missing values")
    X = X.fillna(0)
    print("✓ Missing values filled with 0")
else:
    print("✓ No missing values found")

print(f"✓ Features: {X.shape[1]} symptoms")
print(f"✓ Diseases: {len(y.unique())} unique diseases")

# Encode disease labels
print("\n[3/6] Encoding disease labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"✓ Encoded {len(label_encoder.classes_)} disease labels")

# Split data
print("\n[4/6] Splitting data (80% train, 20% validation)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")

# Train models
models = {}

print("\n[5/6] Training models...")
print("-" * 50)

# 1. Support Vector Machine
print("Training SVM...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_val)
svm_acc = accuracy_score(y_val, svm_pred)
models['svm'] = svm_model
print(f"✓ SVM Accuracy: {svm_acc*100:.2f}%")

# 2. Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
models['random_forest'] = rf_model
print(f"✓ Random Forest Accuracy: {rf_acc*100:.2f}%")

# 3. Naive Bayes
print("Training Naive Bayes...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_val)
nb_acc = accuracy_score(y_val, nb_pred)
models['naive_bayes'] = nb_model
print(f"✓ Naive Bayes Accuracy: {nb_acc*100:.2f}%")

# Test on Testing.csv
print("\n[6/6] Testing on Testing.csv...")
test_data = pd.read_csv('datasets/Testing.csv')
X_test = test_data.drop('prognosis', axis=1)
X_test = X_test.fillna(0)  # Fill missing values in test data too
y_test = label_encoder.transform(test_data['prognosis'])

print(f"Testing data: {len(test_data)} samples")
best_model_name = max(
    [('SVM', svm_acc), ('Random Forest', rf_acc), ('Naive Bayes', nb_acc)],
    key=lambda x: x[1]
)[0]
print(f"Best model: {best_model_name}")

# Save models
print("\n[7/7] Saving models...")
os.makedirs('models', exist_ok=True)

# Save each model
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("✓ Saved: svm_model.pkl")

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ Saved: rf_model.pkl")

with open('models/nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
print("✓ Saved: nb_model.pkl")

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✓ Saved: label_encoder.pkl")

# Save symptom columns
symptom_cols = list(X.columns)
with open('models/symptoms.pkl', 'wb') as f:
    pickle.dump(symptom_cols, f)
print("✓ Saved: symptoms.pkl")

print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"\nModel Accuracies:")
print(f"  SVM:           {svm_acc*100:.2f}%")
print(f"  Random Forest: {rf_acc*100:.2f}%")
print(f"  Naive Bayes:   {nb_acc*100:.2f}%")
print(f"\nAll models saved in 'models/' folder")
print("=" * 50)