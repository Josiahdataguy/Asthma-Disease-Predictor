# Asthma-Disease-Predictor

## Introduction

The Asthma Disease Predictor Model aims to predict the presence of asthma in individuals based on various health indicators. The data for this project was collected from Kaggle.com, which provides a comprehensive dataset with relevant features for building the predictive model. Multiple machine learning algorithms were employed to identify the most effective model for this binary classification task.

## Data Collection

The dataset used for this project was sourced from Kaggle.com, a popular platform for datasets and machine learning competitions. The data includes multiple features relevant to asthma diagnosis, such as demographic information, clinical measurements, and other health indicators. The target variable, "Diagnosis," indicates the presence (1) or absence (0) of asthma.

## Data Preprocessing

Before training the models, the data underwent several preprocessing steps:

- Handling missing values
- Encoding categorical variables
- Normalizing numerical features
- Splitting the data into training and testing sets

The input data excludes the "Diagnosis" column, which serves as the output variable.

## Machine Learning Models Used

Several machine learning models were employed to train and evaluate the performance of the asthma disease predictor. These models include:

1.Logistic Regression
2 Decision Tree
3.Random Forest
4.Support Vector Machine (SVM)
5.k-Nearest Neighbors (k-NN)
6.Gradient Boosting
7.XGBoost
8.Neural Networks

## Model Evaluation

Each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation was employed to ensure the robustness of the evaluation.

## Results
Among all the models tested, the Gradient Boosting Classifier emerged as the most accurate model for predicting asthma. It demonstrated superior performance in terms of accuracy and other evaluation metrics, effectively capturing the complex patterns within the dataset.

## Conclusion
The Asthma Disease Predictor Model showcases the application of various machine learning algorithms to a healthcare-related binary classification problem. By experimenting with different models, Gradient Boosting Classifier was identified as the most effective model for predicting asthma. This project highlights the importance of model selection and evaluation in developing accurate and reliable predictive models.

