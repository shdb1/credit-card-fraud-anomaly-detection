#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing NumPy for numerical operations and handling arrays
import numpy as np 

# Importing pandas for data manipulation and analysis (especially for DataFrames)
import pandas as pd

# Importing scikit-learn (sklearn) for machine learning tools and algorithms
import sklearn

# Importing Matplotlib for data visualization (specifically plotting graphs and charts)
import matplotlib.pyplot as plt

# Importing Seaborn for statistical data visualization (it enhances Matplotlib)
import seaborn as sns

# Importing rcParams from pylab to set the default size of plots
from pylab import rcParams

# Importing classification metrics (like classification report and accuracy score) from sklearn to evaluate model performance
from sklearn.metrics import classification_report, accuracy_score

# Importing Isolation Forest from sklearn (used for anomaly detection, particularly for fraud detection)
from sklearn.ensemble import IsolationForest

# Importing Local Outlier Factor (LOF) from sklearn (another anomaly detection technique)
from sklearn.neighbors import LocalOutlierFactor

# Importing One-Class SVM from sklearn (a machine learning algorithm for anomaly detection)
from sklearn.svm import OneClassSVM


# In[2]:


# Setting the default figure size for all plots to be 14 inches by 8 inches
rcParams['figure.figsize'] = 14, 8 

# Defining a random seed for reproducibility of results (ensures the same results when the code is run multiple times)
RANDOM_SEED = 42

# Defining labels for the classes in the dataset (used to represent 'Normal' and 'Fraud' transactions)
LABELS = ["Normal", "Fraud"]


# In[3]:


# Reading the dataset from a CSV file named 'creditcard.csv' using pandas' read_csv method where separation is a comma.
data = pd.read_csv('creditcard.csv', sep=',')

# Overview of data (first 5 rows only shown by this)
data.head()


# In[4]:


# Displaying a summary of the dataset, including the number of entries, column names,  data types, and the count of non-null values for each column.
data.info()


# ## Exploratory Data Analysis [EDA]

# In[5]:


# Checking if there are any missing (null) values in the dataset.
# The 'isnull()' function returns a DataFrame of the same shape as 'data' with Boolean values (True for nulls, False for non-nulls).
# The 'values.any()' function checks if any True values exist in the DataFrame, indicating the presence of missing values.
data.isnull().values.any()


# In[6]:


# Counting the occurrences of each class (Normal or Fraud) in the 'Class' column
# 'value_counts()' returns the number of occurrences of unique values, sorted in descending order
count_classes = pd.value_counts(data['Class'], sort=True)

# Plotting the class distribution as a bar chart
# 'kind='bar'' specifies a bar plot, 'rot=0' keeps the x-axis labels horizontal
count_classes.plot(kind='bar', rot=0)

# Setting the title of the plot to describe the class distribution of the dataset
plt.title("Transaction Class Distribution")

# Setting the x-axis labels to correspond to the class labels (Normal and Fraud)
plt.xticks(range(2), LABELS)

# Labeling the x-axis as 'Class' to indicate it represents transaction types (Normal/Fraud)
plt.xlabel("Class")

# Labeling the y-axis as 'Frequency' to indicate the count of each class
plt.ylabel("Frequency")


# In[7]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[8]:


print(fraud.shape,normal.shape)


# In[9]:


## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in transaction class Fraud?
fraud.Amount.describe()


# In[10]:


## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in transaction class Normal?
normal.Amount.describe()


# In[11]:


# Creating a figure with two vertically stacked subplots that share the same x-axis(Histogram Amount Vs Transaction)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Setting a common title for the entire figure
f.suptitle('Amount per transaction by class')

# Defining the number of bins to be used in the histograms
bins = 50

# Plotting the distribution of transaction amounts for fraudulent transactions in the first subplot
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')  # Title for the first subplot

# Plotting the distribution of transaction amounts for normal transactions in the second subplot
ax2.hist(normal.Amount, bins=bins)
ax2.set_title('Normal')  # Title for the second subplot

# Setting the label for the x-axis (shared across both subplots)
plt.xlabel('Amount ($)')

# Setting the label for the y-axis (number of transactions per bin)
plt.ylabel('Number of Transactions')

# Limiting the x-axis to a range between 0 and 25,000 for better visualization
plt.xlim((0, 25000))

# Setting a logarithmic scale for the y-axis to better visualize differences in counts (especially for imbalanced data)
plt.yscale('log')

# Displaying the plot
plt.show()


# In[12]:


# We will investigate whether fraudulent transactions occur more frequently during specific time intervals.(Scatter plot Amount vs Time)
# A visual comparison between fraudulent and normal transactions will help identify any time-based patterns.

# Creating a figure with two subplots (2 rows, 1 column) that share the same x-axis (Time)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Setting a common title for the entire figure
f.suptitle('Time of transaction vs Amount by class')

# Plotting a scatter plot for fraudulent transactions
# x-axis: Time of transaction, y-axis: Amount of transaction
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')  # Title for the fraud subplot

# Plotting a scatter plot for normal transactions
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')  # Title for the normal transactions subplot

# Labeling the shared x-axis (applies to both subplots)
plt.xlabel('Time (in Seconds)')

# Labeling the y-axis to indicate the transaction amount
plt.ylabel('Amount')

# Displaying the plots
plt.show()


# In[13]:


# Taking a random 10% sample of the entire dataset using pandas' sample() method.
# 'frac=0.1' specifies that 10% of the data should be sampled.
# 'random_state=1' ensures reproducibility—every time you run the code, you get the same sample.
sample_data = data.sample(frac=0.1, random_state=1)

# Displaying the shape (number of rows and columns) of the sampled dataset
sample_data.shape


# In[14]:


# complete data
data.shape


# In[15]:


# Filtering the dataset to get only the fraudulent transactions (where Class == 1)
bad_transactions = sample_data[sample_data['Class'] == 1]

# Filtering the dataset to get only the valid (non-fraudulent) transactions (where Class == 0)
valid_transactions = sample_data[sample_data['Class'] == 0]

# Calculating the ratio of fraudulent transactions to valid transactions
# This ratio (outlier_fraction) is useful in understanding class imbalance,
# and is often used as an input parameter for anomaly detection algorithms
outlier_fraction = len(bad_transactions) / float(len(valid_transactions))

# Printing the calculated ratio of fraud to valid transactions
print('Ratio of fraud to valid transactions: ', outlier_fraction)

# Displaying the number of fraudulent transactions in the sample
print("Fraud Cases : {}".format(len(bad_transactions)))

# Displaying the number of valid (non-fraudulent) transactions in the sample
print("Valid Cases : {}".format(len(valid_transactions)))


# In[16]:


## Correlation
# Calculating the correlation matrix for the dataset
# This shows how strongly each feature is linearly related to the others
corrmat = sample_data.corr()

# Getting the column names (feature names) from the correlation matrix
# This will be used to access all features for plotting the heatmap
top_corr_features = corrmat.index
corrmat.to_csv("correlation_matrix.csv")
# Setting the figure size for the heatmap plot (20x20 inches)
plt.figure(figsize=(30, 30))

# Plotting the heatmap using Seaborn
# - 'data[top_corr_features].corr()' ensures the heatmap is based on the same correlation matrix
# - 'annot=True' displays the correlation values in each cell
# - 'cmap="RdYlGn"' sets the color theme (Red-Yellow-Green) to indicate strength and direction of correlation
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")


# In[17]:


# Create independent and dependent features (input features and the target variable)

# Get a list of all column names in the dataset
columns = sample_data.columns.tolist()

# Filter the columns to exclude the target variable 'Class'
# These remaining columns will be used as input features (independent variables)
columns = [c for c in columns if c not in ["Class"]]

# Store the name of the target variable we are trying to predict
target = "Class"

# Define a random state generator with a fixed seed for reproducibility
state = np.random.RandomState(42)

# Assign the input features (X) by selecting all columns except 'Class'
X = sample_data[columns]

# Assign the target labels (Y), which is the 'Class' column
Y = sample_data[target]

# Generate a random matrix with the same shape as X to simulate outlier data
# This is often used for testing or comparison in anomaly detection
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

# Print the shape (rows, columns) of the feature matrix X
print(X.shape)

# Print the shape of the target vector Y
print(Y.shape)


# In[18]:


## Define the outlier detection methods

# Creating a dictionary to store the different anomaly detection algorithms
# Each key in the dictionary represents the name of the algorithm, and the value is an instance of the corresponding model
classifiers = {
    # Isolation Forest Algorithm (with updated behaviour to suppress future warning)
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=len(X),  # number of trees and maximum samples to train each tree
                                       contamination=outlier_fraction,  # proportion of outliers in the data
                                       random_state=state,  # ensures reproducibility of results by fixing the random seed
                                       behaviour='new',  # updated behaviour to align with newer scikit-learn API
                                       verbose=0),  # suppresses verbose output during training
    
    # Local Outlier Factor Algorithm
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, algorithm='auto',  # number of neighbors and algorithm type for calculation
                                              leaf_size=30,  # leaf size for the tree-based algorithms
                                              metric='minkowski',  # distance metric used to compute the nearest neighbors
                                              p=2,  # p=2 corresponds to Euclidean distance
                                              metric_params=None,  # additional parameters for the distance metric (not used here)
                                              contamination=outlier_fraction),  # proportion of outliers in the data
    
    # Support Vector Machine (One-Class SVM) Algorithm
    "Support Vector Machine": OneClassSVM(kernel='rbf', degree=3, gamma=0.1, nu=0.05,  # kernel type, polynomial degree, and gamma parameter for RBF kernel
                                         max_iter=-1)  # maximum number of iterations (-1 means no limit)
}


# In[19]:


n_outliers = len(bad_transactions)

# Looping over each classifier (anomaly detection method) in the 'classifiers' dictionary
for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # Fit the classifier to the data and tag outliers (fraudulent transactions)
    print("classifier name: "+clf_name)
    # For Local Outlier Factor (LOF), we use 'fit_predict' to compute the outlier predictions directly
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)  # Predictions: 1 for inliers (valid), -1 for outliers (fraud)
        scores_prediction = clf.negative_outlier_factor_  # LOF also provides outlier scores
    
    # For Support Vector Machine (One-Class SVM), we fit the model and use the 'predict' method to classify outliers
    elif clf_name == "Support Vector Machine":
        clf.fit(X)  # Fit the SVM model on the data
        y_pred = clf.predict(X)  # Predict the class (1 for outlier, -1 for inlier)
    
    # For Isolation Forest and other classifiers, we use the 'fit' method to train and 'decision_function' to compute outlier scores
    else:
        clf.fit(X)  # Fit the model to the data
        scores_prediction = clf.decision_function(X)  # Get outlier scores (higher values mean normal)
        y_pred = clf.predict(X)  # Predict the class (1 for outlier, 0 for inlier)

    # Convert the predictions to 0 for valid transactions and 1 for fraudulent transactions
    y_pred[y_pred == 1] = 0  # Change 1 to 0 (valid transactions)
    y_pred[y_pred == -1] = 1  # Change -1 to 1 (fraudulent transactions)

    # Calculate the number of errors (misclassifications)
    n_errors = (y_pred != Y).sum()

    # Print the results for each classifier
    print("{}: {}".format(clf_name, n_errors))  # Output the number of errors for each classifier
    print("Accuracy Score :")
    print(accuracy_score(Y, y_pred))  # Print the accuracy score
    print("Classification Report :")
    print(classification_report(Y, y_pred))  # Print detailed classification metrics (precision, recall, f1-score)


# In[ ]:




