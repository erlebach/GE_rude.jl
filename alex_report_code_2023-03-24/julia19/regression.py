# Create a code to perform a linear regression analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create a function to perform a linear regression analysis
def linear_regression(x, y):
    # Perform linear regression analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Print the results
    print('slope: %f, intercept: %f, r_value: %f, p_value: %f, std_err: %f' % (slope, intercept, r_value, p_value, std_err))
    # Plot the results
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, intercept + slope * x, 'r', label='fitted line')
    # add a plot title
    plt.title("Linear Regression")
    plt.legend()
    plt.show()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
linear_regression(x, y)

# Output
# slope: 2.000000, intercept: 0.000000, r_value: 1.000000, p_value: 0.000000, std_err: 0.000000

# Duplicate the above code using sklearn
# Path: regression_sklearn.py
# Create a code to perform a linear regression analysis
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

# Create a function to perform a linear regression analysis
def linear_regression(x, y):
    # Perform linear regression analysis
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    # Print the results
    print('slope: %f, intercept: %f' % (regr.coef_, regr.intercept_))
    # Plot the results
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, regr.intercept_ + regr.coef_ * x, 'r', label='fitted line')
    plt.legend()
    plt.show()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# linear_regression(x, y)
print("linear_regression(x, y) = ", linear_regression(x, y))

# repeat the above code using torch
# Path: regression_torch.py
# Create a code to perform a linear regression analysis
import torch
import torch.nn as nn

# Create a function to perform a linear regression analysis
def linear_regression_torch(x, y):
    # Perform linear regression analysis
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1000):
        inputs = torch.from_numpy(x).float()
        labels = torch.from_numpy(y).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Print the results
    print('slope: %f, intercept: %f' % (model.weight.item(), model.bias.item()))
    # Plot the results
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, model.bias.item() + model.weight.item() * x, 'r', label='fitted line')
    # title
    plt.title("Linear Regression with torch")
    plt.legend()
    plt.show()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
linear_regression_torch(x, y)
print("linear_regression_torch(x, y) = ", linear_regression_torch(x, y))

