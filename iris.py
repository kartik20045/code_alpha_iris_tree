# iris_csv_classification.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('iris.csv')

# Optional: Drop ID column if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Step 2: Preview the dataset
print("Dataset Preview:")
print(df.head())

# Step 3: Visualize the dataset
sns.pairplot(df, hue='Species')
plt.suptitle("Iris Dataset - Pairplot", y=1.02)
plt.show()

# Step 4: Prepare features and target
X = df.drop(columns=['Species'])
y = df['Species']

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
