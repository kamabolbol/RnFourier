import torch
import torch.nn as nn
import numpy as np

# Time vector that will be used as input of our NN
t_numpy = np.arange(0, 5.01, 0.01, dtype=np.float32)

# Neural Network Definition
class NeuralNet(nn.Module):
    def __init__(self, hidden_size, output_size=1, input_size=1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.LeakyReLU()
        self.l4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        return out

# Convert numpy array to torch tensor and set requires_grad
t_train = torch.from_numpy(t_numpy).reshape(len(t_numpy), 1)
t_train.requires_grad_(True)

# Constant for the model
k = 1

# Instantiate the model with 50 neurons in the hidden layers
model = NeuralNet(hidden_size=50)

# Loss and optimizer
learning_rate = 8e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Loss for the differential equation
initial_condition_loss = nn.MSELoss()  # Loss for the initial condition

# Number of epochs
num_epochs = int(1e4)

# Training loop
for epoch in range(num_epochs):
    # Randomly perturbing the training points to have a wider range of times
    epsilon = torch.normal(0, 0.1, size=(len(t_train), 1)).float()
    t_perturbed = t_train + epsilon

    # Forward pass
    y_pred = model(t_train)

    # Calculate the derivative of the forward pass w.r.t. the input (t)
    dy_dt = torch.autograd.grad(
        y_pred,
        t_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # Define the differential equation and calculate the loss
    loss_DE = criterion(dy_dt + k * y_pred, torch.zeros_like(dy_dt))

    # Define the initial condition loss (y(0) = 1)
    loss_IC = initial_condition_loss(model(torch.tensor([[0.0]])), torch.tensor([[1.0]]))

    # Total loss
    loss = loss_DE + loss_IC

    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")
