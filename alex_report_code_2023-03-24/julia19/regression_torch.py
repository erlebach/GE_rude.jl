import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Create a function to perform a linear regression analysiing torch
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
        print(model.weight.item())
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

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1,1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(-1,1)
# linear_regression_torch(x, y)
print("linear_regression_torch(x, y) = ", linear_regression_torch(x, y))

# the following is an error: 
"""
File "/Users/erlebach/opt/anaconda3/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x10 and 1x1)
"""

# Create a function to perform a linear regression analysiing torch
# Use 20 points and a batch size of 5
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
        print(model.weight.item())
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

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).reshape(-1,1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]).reshape(-1,1)
linear_regression_torch(x, y)
print("linear_regression_torch(x, y) = ", linear_regression_torch(x, y))
