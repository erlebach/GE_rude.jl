using Random
using Flux: Chain, Dense, Optimiser, ExpDecay,  train!, params, predict, mse
using Optimisers: SGD

function linear_regression(x, y, lr=0.01, epochs=100, batch_size=1)
    # Convert the input data and output data to tensors
    x = reshape(x, :, 1)
    y = reshape(y, :, 1)

    # Define the model
    model = Chain(Dense(1, 1))

    # Define the loss function
    loss(y_pred, y_true) = mse(y_pred, y_true)

    # Define the optimizer
    optimizer = Optimiser(ExpDecay(lr), SGD())

    # Train the model
    losses = Float32[]
    for epoch in 1:epochs
        for i in 1:batch_size:size(x, 1)
            x_batch = x[i:i+batch_size-1,:]
            y_batch = y[i:i+batch_size-1,:]

            Flux.train!(loss, params(model), [(x_batch, y_batch)], optimizer)
        end
        push!(losses, loss(predict(model, x), y))
    end

    # Return the trained model and the losses
    return model, losses
end

# Generate some sample input and output data
Random.seed!(123)
x = collect(range(0, stop=1, length=20))
y = 2*x .+ 1 .+ randn(20)*0.1

# Perform linear regression with batch size 5
model, losses = linear_regression(x, y, lr=0.01, epochs=100, batch_size=5)

# Print the learned parameters
println("Weight: $(model.layers[1].W[1])")
println("Bias: $(model.layers[1].b[1])")

# Convert x to a tensor for plotting
x_tensor = reshape(x, :, 1)

# Plot the data points and the learned line
using Plots
scatter(x, y, label="Data")
plot!(x, predict(model, x_tensor), label="Learned Line", color="red")
xlabel!("x")
ylabel!("y")
legend!()

# Calculate R-squared
y_pred = predict(model, x_tensor)
r_squared = 1 - sum((y .- y_pred).^2) / sum((y .- mean(y)).^2)
println("R-squared: $r_squared")

# Plot the loss curve
plot(losses, xlabel="Epoch", ylabel="Loss")