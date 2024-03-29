#!/usr/bin/env python
# coding: utf-8

# In[16]:


# 1. Import necessary libraries
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 2. Load data and basic exploration
fish_data_path = r"C:\Users\Siddharth\Documents\Lab4\Fish market dataset"

def load_fish_data():
    csv_path = os.path.join(fish_data_path, 'Fish.csv')
    return pd.read_csv(csv_path)

fish_data = load_fish_data()

# 3. Handle outliers
def remove_outliers_iqr(fish_data, features):
    Q1 = fish_data[features].quantile(0.25)
    Q3 = fish_data[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return fish_data[~((fish_data[features] < lower_bound) | (fish_data[features] > upper_bound)).any(axis=1)]

features_to_check = ['Length1', 'Height', 'Width', 'Weight']  # Remove 'Length2' and 'Length3'
fish_data_train_no_outliers_iqr = remove_outliers_iqr(fish_data, features_to_check)

fish_data = fish_data_train_no_outliers_iqr

# 4. Visualize the data

fish_data.boxplot(figsize=(18, 10))
plt.show()

# 5. Preprocess data
X = fish_data.drop(['Species', 'Length2', 'Length3'], axis=1)  # Drop 'Length2' and 'Length3'
y = fish_data['Species']
print(X.columns)
fish_data['Species'] = fish_data['Species'].replace(fish_data['Species'].unique(), np.arange(len(fish_data['Species'].unique())))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6. Train the model
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

num_attribs = list(X)
cat_attribs = []

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs)
])

x_train_prepared = full_pipeline.fit_transform(x_train)
x_test_prepared = full_pipeline.transform(x_test)
print(x_test_prepared)
classifier = LogisticRegression(random_state=42)
classifier.fit(x_train_prepared, y_train)

# 7. Evaluate the model
y_pred = classifier.predict(x_test_prepared)
cm_lg = confusion_matrix(y_test, y_pred)
ac_lg = accuracy_score(y_test, y_pred)

print("Confusion Matrix for Logistic Regression:")
print(cm_lg)
print("Accuracy Score for Logistic Regression:", ac_lg)


# Save the model
pickle.dump(classifier, open('model.pkl', 'wb'))

# Load the model and make predictions
model1 = pickle.load(open('model.pkl', 'rb'))
predicted_species = model1.predict([[242,23.2, 11.5200, 4.0200]])  # Using example data without 'Length2' and 'Length3'
print("Predicted Species:", predicted_species)


# In[ ]:




