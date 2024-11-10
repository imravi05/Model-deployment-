# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("D:\Machine Learnig NInja\Coffee Quality\synthetic_coffee_quality_dataset.csv")

# Select features and preprocess them
features = ['acidity', 'caffeine', 'aroma', 'texture', 'bitterness', 'sweetness', 'body', 'aftertaste', 'color']
X = df[features]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode roast level and target variable
le = LabelEncoder()
df['roast_level_encoded'] = le.fit_transform(df['roast_level'])
X_with_roast = pd.DataFrame(X, columns=features)
X_with_roast['roast_level'] = df['roast_level_encoded']
y = df['quality_label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_with_roast, y, test_size=0.2, random_state=42)

# Train model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the model and the scaler
joblib.dump(classifier, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model, scaler, and encoder saved successfully.")
