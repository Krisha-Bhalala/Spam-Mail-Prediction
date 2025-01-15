# Spam Mail Prediction

This project predicts whether an email is **spam** or **ham (non-spam)** using a **Logistic Regression** machine learning model. The model is trained using a dataset of labeled emails and employs **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.

## Project Overview

The goal of this project is to predict whether an email is spam or not based on its content. This is achieved by preprocessing the email data, applying machine learning techniques for text classification, and evaluating the model's performance.

### Steps Involved:
1. **Data Preprocessing**: Load the dataset, clean the data, and handle any missing values.
2. **Label Encoding**: Convert spam mails to `0` and ham mails to `1`.
3. **Feature Extraction**: Use **TF-IDF** to convert the text data into numerical format suitable for training the machine learning model.
4. **Model Training**: Use **Logistic Regression** to train the model on the training data.
5. **Model Evaluation**: Test the model on unseen data and calculate its accuracy on both training and testing datasets.
6. **Prediction System**: Predict whether a given email is spam or ham.

## Requirements

Ensure you have the following libraries installed:
- numpy
- pandas
- scikit-learn
- google-colab(only if you're using Google Colab for running the notebook)

## Dataset

The dataset for this project contains a collection of emails categorized as spam and ham. You can upload your own dataset or use the provided sample (mails_data.csv). The dataset must contain at least the following two columns:

Message: The content of the email.
Category: The label for the email (spam or ham).

## How to Run

Clone the repository:
git clone https://github.com/Krisha-Bhalala/Spam-Mail-Prediction.git

Navigate to the project directory:
cd Spam-Mail-Prediction

Download the dataset: Make sure the mails_data.csv file is placed in the same directory as the notebook or modify the file path in the script.

Run the notebook: Open Spam Mail Prediction.ipynb in Jupyter Notebook or Google Colab and run the cells in sequence. The model will be trained and evaluated, and you will see predictions for test data.

Run a sample prediction: You can enter a custom email message in the input_mail variable and the model will predict whether it's spam or ham.

Example:

input_mail = ["Nah I don't think he goes to usf, he lives around here though"]

## How It Works

- **Data Preprocessing**: The dataset is loaded, and missing values are handled by replacing them with empty strings.
- **Label Encoding**: The categories spam and ham are encoded as 0 and 1, respectively.
- **Feature Extraction**: The text data is converted into numerical features using the TF-IDF Vectorizer.
- **Training the Model**: Logistic Regression is trained on the processed data.
- **Model Evaluation**: The model's accuracy is evaluated on both training and testing data.
- **Prediction**: For any new input email, the model predicts whether the email is spam or ham.

## Results

- Accuracy on Training Data: The model's accuracy on the training set is printed after the training phase.
- Accuracy on Test Data: The model’s performance is then evaluated on the test set, and the accuracy is displayed.
- Prediction on Input Data: The model predicts whether a sample email is spam or ham.

### Example Output

- Accuracy on training data = 0.98
- Accuracy on test data = 0.95
- Prediction is: [0]
- It is a spam mail!!

## License

This project is licensed under the MIT License.
