# Import necessary libraries
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv(r"C:\Users\mylap\A VS CODE\Salary_Data.csv")
# Check the shape of the dataset
print("Dataset Shape:", dataset.shape) # (30, 2)

# Feature selection (independent variable X and dependent variable y)
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Reshape x_train and x_test into 2D arrays if they are single feature columns
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# You don't need to reshape y_train, as it's the target variable
# Fit the Linear Regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the results for the test set
y_pred = regressor.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 12 + c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp = m_slope * 20 + c_intercept
print(pred_20yr_emp_exp)

# Optional: Output the coefficients of the linear model
print(f"Intercept: {regressor.intercept_}")
print(f"Coefficient: {regressor.coef_}")

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# stats for ml 

# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

#STATISTICS FOR MACHINE LEARNING
dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset['Salary'].mode()

dataset.describe()

dataset.var()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()

dataset.corr()


# ssr 
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#sse
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#sst 
mean_total = np.mean(dataset.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2 
r_square = 1 - SSR/SST
print(r_square)


bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)


# deployment in flask & HTML
# mlops (azur, googlecolab, heroku, kubernate)

import pickle

# Save the trained model to disk
filename = 'linear_regression_model.pkl'

# Open a file in wrire-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
    
    
print("Model has been pickled and saved as linear_regression_model.pkl")    
    
