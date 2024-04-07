import urllib.request
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/diabetes.csv"

# Download the file
file_path = "diabetes.csv"
urllib.request.urlretrieve(url, file_path)

# Assuming the file is downloaded correctly
diabetes = pd.read_csv(file_path)
diabetes.head()

from azureml.core import Run
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get experiment run context
run = Run.get_context()

# Prepare dataset
feature_cols = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'Age']
label_col = ['Diabetic']

X, y = diabetes[feature_cols].values, diabetes[label_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training Set size: {len(X_train)}\nTest Set size = {len(X_test)}\n\n")

# Train the model
reg= 0.1
model = LogisticRegression(C=1/reg, solver='liblinear')
model.fit(X_train, y_train.flatten())

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', float(acc))

# Save the trained model
import os
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()