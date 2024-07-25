# -*- coding: utf-8 -*-
"""MPML_FIXMODEL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VbjEMZ-2PtN3pvZsKCSDR4ujvAd4Al9g
"""

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

onlinefood_data= pd.read_csv('/content/onlinefoods.csv')
onlinefood_data.head()

# Menghapus Kolom
data_cleaned = onlinefood_data.drop(columns=['Unnamed: 12'])

data_cleaned.head()

# Check for missing values
missing_values = data_cleaned.isnull().sum()
missing_values

# Perform one-hot encoding on categorical variables
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the category columns
category_cols = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income',
                 'Educational Qualifications', 'Feedback', 'Output']

# Initialize LabelEncoder
labelEncoder = LabelEncoder()
mapping_dict = {}

# Apply LabelEncoder to each category column and store the mappings
for col in category_cols:
    data_cleaned[col] = labelEncoder.fit_transform(data_cleaned[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col] = le_name_mapping

# Display the mapping dictionary
print(mapping_dict)

# Display the first few rows of the encoded dataframe
print(data_cleaned.head())

# Memisahkan fitur dan target
X = data_cleaned.drop(columns=['Output'])
y = data_cleaned['Output']

# Membagi dataset menjadi data pelatihan (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#MODEL KNN

# Melatih model KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

# Hyperparameter tuning untuk KNN
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Model terbaik
best_knn = grid_search.best_estimator_

# Melakukan prediksi pada data uji
predictions_knn = model_knn.predict(X_test)

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_test, predictions_knn)
precision = precision_score(y_test, predictions_knn,average='weighted')
recall = recall_score(y_test, predictions_knn,average='weighted')
f1 = f1_score(y_test, predictions_knn,average='weighted')
conf_matrix = confusion_matrix(y_test, predictions_knn)

# Validasi Silang
cv_scores = cross_val_score(best_knn, X, y, cv=5, scoring='accuracy')

print(f"KNN Akurasi Prediksi: {accuracy}")
print(f"KNN Precision: {precision}")
print(f"KNN Recall: {recall}")
print(f"KNN F1 Score: {f1}")
print("KNN Confusion Matrix:")
print(conf_matrix)
print("KNN Cross-Validation Scores:", cv_scores)
print("KNN Mean Cross-Validation Score:", cv_scores.mean())
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)