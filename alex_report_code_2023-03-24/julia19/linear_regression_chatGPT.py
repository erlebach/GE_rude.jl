import matplotlib.pyplot as plt
import numpy as np
import torch

def linear_regression(x, y, lr=0.01, epochs=100, batch_size=1):
    # Convert the input data and output data to PyTorch tensors
    x = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    # Define the model
    model = torch.nn.Linear(1, 1)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Train the model
    losses = []
    for epoch in range(epochs):
        for i in range(0, len(x), batch_size):
            # Get a batch of input data and output data
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)

            # Compute the loss
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Append the loss after each epoch
        losses.append(loss.item())

    # Return the trained model and the losses
    return model, losses

# Generate some sample input and output data
x = np.linspace(0, 1, 20)
y = 2*x + 1 + np.random.randn(20)*0.1

# Perform linear regression with batch size 5
model, losses = linear_regression(x, y, lr=0.01, epochs=100, batch_size=5)

# Print the learned parameters
print('Weight: {:.2f}, Bias: {:.2f}'.format(model.weight.item(), model.bias.item()))

# Convert x to a tensor for plotting
x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)

# Plot the data points and the learned line
plt.scatter(x, y, label='Data')
plt.plot(x, model(x_tensor).detach().numpy().flatten(), label='Learned Line', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Calculate R-squared
y_pred = model(x_tensor).detach().numpy().flatten()
r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print('R-squared:', r_squared)

# Plot the loss curve
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()