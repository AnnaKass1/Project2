## Credit risk assessment

## Credit Risk Analysis and Clustering

#Purpose

This project demonstrates credit risk analysis and clustering using machine learning techniques. It utilizes the credit risk dataset to train and evaluate models for predicting loan status and performs exploratory data analysis and clustering to identify patterns in the data. The code is designed to be executed in Google Colab, an online Jupyter notebook environment.

#Techinical Goals

1. K-nearest neighbors (KNN) Classifier: The code trains a KNN classifier on the scaled data. The goal is to explore an alternative        machine learning algorithm and evaluate its performance in predicting loan status
2. Model Training: The code trains a logistic regression model using the original training data (X_train, y_train) and the oversampled      training data (X_oversampled, y_oversampled). The goal is to fit the logistic regression model to the training data and learn the        coefficients that minimize the difference between predicted probabilities and actual loan statuses

##Dataset 
Retrived simiualted dataset from Kaggle

The code consists of several modules and steps:

1. Importing necessary modules and libraries: This section imports the required Python modules and libraries for data analysis, visualization, and machine learning.

2. Reading the credit risk dataset: The code reads the credit risk dataset from a local file called "credit_risk_dataset.csv" using the Pandas library. The dataset contains information about loan applications, including various features such as loan amount, interest rate, employment length, home ownership status, and loan status.

3. Preprocessing the data: This step involves cleaning and preprocessing the dataset. It includes dropping unnecessary columns, handling missing values, and encoding categorical variables.

4. Training a logistic regression model: The code splits the preprocessed data into training and testing sets using the train_test_split function from scikit-learn. It then trains a logistic regression model on the training data to predict the loan status.

5. Evaluating model performance: The code generates predictions using the trained logistic regression model on both the training and testing data. It evaluates the model's performance by calculating the accuracy score, creating a confusion matrix, and generating a classification report for the testing data.

6. Resampling the training data using RandomOverSampler: To address class imbalance, the code utilizes the RandomOverSampler module from the imbalanced-learn library. It resamples the training data to create a balanced dataset with equal representation of each class.

7. Training a logistic regression model with resampled data: The resampled training data is used to train another logistic regression model. This model aims to provide improved performance by considering the balanced class distribution.

8. Evaluating model performance on resampled data: The code generates predictions using the logistic regression model trained on the resampled data. It evaluates the model's performance by calculating the accuracy score, creating a confusion matrix, and generating a classification report for the resampled testing data.

9. Exploratory data analysis and clustering: This section explores the relationships between different variables in the dataset. It includes creating a correlation heatmap to visualize the pairwise correlations between numerical features. Additionally, a boxplot is generated to show the distribution of loan amounts grouped by loan purpose. The code also performs clustering using the K-means algorithm to identify patterns and similarities among the data points.

10. Encoding categorical variables and feature scaling: The code encodes the categorical variables using one-hot encoding (get_dummies) and performs feature scaling using StandardScaler. Feature scaling ensures that all features are on the same scale, which can improve the performance of machine learning models.

11. Training a K-nearest neighbors (KNN) classifier: The code trains a K-nearest neighbors (KNN) classifier on the scaled data. The KNN model classifies new data points based on the class labels of their k nearest neighbors in the training data.

12. Evaluating KNN model performance: The code generates predictions using the trained KNN classifier on the testing data. It evaluates the model's performance by calculating the accuracy score and generating a classification report.
