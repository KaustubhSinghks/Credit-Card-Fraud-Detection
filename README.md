Name: KAUSTUBH SINGH

Company : CODTECH IT SOLUTIONS

ID : CT8ML1185

Domain : MACHINE LEARNING

Duration : June to August 2024

Mentor : NEELAM HARISH


# Credit-Card-Fraud-Detection
**Project Overview**
The Credit Card Fraud Detection project aims to build a machine learning model to detect fraudulent credit card transactions using a Support Vector Classifier (SVC). The project involves preprocessing the data, training a model, and evaluating its performance.

**Objectives**
Develop a Fraud Detection Model: Create a model that can accurately classify credit card transactions as either fraudulent or legitimate.
Evaluate Model Performance: Assess the accuracy of the model using various metrics and improve its performance if necessary.


**Key Activities**
Data Collection and Loading:

Load the credit card transaction dataset from a CSV file using Pandas.
Understand the dataset structure and features.
Data Preprocessing:

Extract features (X) and labels (y) from the dataset.
Split the data into training and testing sets using train_test_split.
Standardize the feature values using StandardScaler to ensure uniform scale for the SVC model.
Model Training:

Initialize and train the Support Vector Classifier (SVC) with a radial basis function (RBF) kernel.
Fit the model on the standardized training data.
Model Evaluation:

Predict the class labels for the test set.
Calculate the accuracy of the model using accuracy_score to determine how well the model performs on unseen data.
Result Output:

Print the accuracy score of the model to evaluate its performance.


**Technologies Used**
Python: Programming language used for implementation.
Pandas: Library for data manipulation and analysis.
NumPy: Library for numerical operations.
Scikit-learn: Library for machine learning, including model training and evaluation.
Requirements
Python 3.x
Pandas
NumPy
Scikit-learn
You can install the required libraries using pip:

bash
Copy code
pip install pandas numpy scikit-learn
How to Run
Prepare the Dataset: Ensure the dataset file creditcard.csv is available in the working directory.

Execute the Script: Run the Python script using the following command:

bash
Copy code
python your_script_name.py
View the Results: The accuracy of the model will be printed to the console.

**Code Explanation**
Data Loading: Reads the dataset and extracts features and labels.
Data Preprocessing: Splits the data into training and testing sets, and standardizes the features.
Model Training: Trains an SVC model with an RBF kernel on the training data.
Model Evaluation: Predicts the labels for the test set and calculates the accuracy of the model.
