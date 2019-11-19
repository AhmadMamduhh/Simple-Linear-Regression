# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values.reshape(30,1)
y = dataset.iloc[:, 1].values.reshape(30,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Standardization of The Datasets
"""standard_dev= np.std(X_train)
mean_value= np.mean(X_train)
X_train = (X_train - mean_value) / standard_dev
X_test = (X_test - mean_value) / standard_dev """

# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor=regressor.fit(X_train,y_train)

# Predicting Sets
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# Data Visualization
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,y_train_pred,c='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.Figure
plt.scatter(X_test,y_test,c='red')
plt.plot(X_test,y_test_pred,c='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Number of training examples
m_train = X_train.shape[0]

# Number of test examples
m_test = X_test.shape[0]

# Average Squared Error of Training Set
cost_train = np.sum(np.power(y_train - y_train_pred,2)) * (1/m_train)
print('The average cost of the training set: ' + str(cost_train))
 
# Average Squared Error of Test Set
cost_test = np.sum(np.power(y_test - y_test_pred,2)) * (1/m_test)
print('The average cost of the test set: ' + str(cost_test))