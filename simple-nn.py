import torch
import torch.nn as nn
import torch.optim as optim


# ensure results are reproducible and consistent every time
torch.manual_seed(42)

# distances in miles for last four deliveries
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# corresponding delivery time in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# create a model with one input and one output
model = nn.Sequential(nn.Linear(1, 1))

# loss function
loss_function = nn.MSELoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train
for epoch in range(500):
    # reset the optimizer's gradients
    optimizer.zero_grad()
    # make predictions (forward pass)
    outputs = model(distances)
    # calculate the loss
    loss = loss_function(outputs, times)
    # calculate adjustments (backward pass)
    loss.backward()
    # update the model's parameters
    optimizer.step()
    # print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
