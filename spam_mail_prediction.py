# -*- coding: utf-8 -*-
"""Spam Mail Prediction"""

# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and pre-processing

# Load the dataset containing email data
raw_mails_data = pd.read_csv('/content/mails_data.csv')

# Mount Google Drive to access files (for use in Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Display the raw dataset
print(raw_mails_data)

# Replace null values with an empty string to handle missing data
mails_data = raw_mails_data.where((pd.notnull(raw_mails_data)), '')

# Display the first 5 rows of the dataframe to verify data loading
mails_data.head()

# Check the number of rows and columns in the dataset
mails_data.shape

# Label Encoding

# Label spam emails as 0 and non-spam (ham) emails as 1
mails_data.loc[mails_data['Category'] == 'spam', 'Category'] = 0
mails_data.loc[mails_data['Category'] == 'ham', 'Category'] = 1

# Separate the dataset into text (X) and labels (Y)
x = mails_data['Message']
y = mails_data['Category']

# Print the text data (X) and labels (Y) for verification
print(x)
print(y)

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Print shapes of the datasets for confirmation
print(x.shape)  # Total dataset
print(x_train.shape)  # Training set
print(x_test.shape)  # Test set

# Feature Extraction

# Use TfidfVectorizer to convert text data into numerical features
# min_df=1 ensures that terms with at least 1 occurrence are considered
# stop_words='english' removes common stop words, and lowercase=True converts text to lowercase
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Transform the training and test text data into feature vectors
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# Print transformed feature vectors for verification
print("Transformed x_train_features:", x_train_features)
print("Transformed x_test_features:", x_test_features)

# Model Training

# Initialize the logistic regression model
model = LogisticRegression()

# Print unique values of y_train to ensure labels are correctly encoded
print(y_train)
print(y_train.unique() if hasattr(y_train, "unique") else set(y_train))

# Convert y_train to integers for compatibility with the logistic regression model
y_train = y_train.astype(int)

# Check the data type of y_train after conversion
print(y_train.dtypes)

# Train the logistic regression model with the training data
model.fit(x_train_features, y_train)

# Model Evaluation on Training Data

# Predict labels for the training data
prediction_on_training_data = model.predict(x_train_features)

# Calculate accuracy on the training data
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Accuracy on training data =', accuracy_on_training_data)

# Model Evaluation on Test Data

# Predict labels for the test data
prediction_on_test_data = model.predict(x_test_features)

# Convert y_test to integers for compatibility
y_test = y_test.astype(int)

# Calculate accuracy on the test data
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Accuracy on test data =', accuracy_on_test_data)

# Building a Predictive System

# Example input email to test the predictive system
input_mail = ["Nah I don't think he goes to usf, he lives around here though"]
print(input_mail)

# Convert the input email into numerical features using TfidfVectorizer
input_data_features = feature_extraction.transform(input_mail)

# Make a prediction for the input email
prediction = model.predict(input_data_features)
print('Prediction is:', prediction)

# Output the prediction result
if(prediction[0] == 1):
    print('It is a ham mail!!')  # Non-spam email
else:
    print('It is a spam mail!!')  # Spam email
