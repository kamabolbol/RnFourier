import torch
import torch.nn as nn
import numpy as np

# Définition du réseau de neurones
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

# Vecteur temporel
t_numpy = np.arange(0, 5 + 0.01, 0.01, dtype=np.float32)
t = torch.from_numpy(t_numpy).reshape(len(t_numpy), 1)
t.requires_grad_(True)

# Constante du modèle
k = 1.0

# Instanciation du modèle
model = NeuralNet(hidden_size=50)

# Fonction de perte (MSE)
criterion = nn.MSELoss()

# Optimiseur
learning_rate = 8e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Fonction de perte pour la condition initiale
def initial_condition_loss(y_pred, y_true):
    return criterion(y_pred, y_true)

# Entraînement
num_epochs = int(1e4)
for epoch in range(num_epochs):
    # Perturbation aléatoire
    epsilon = torch.normal(0, 0.1, size=(len(t), 1)).float()
    t_train = t + epsilon

    # Forward pass
    y_pred = model(t_train)

    # Calcul de la dérivée dy/dt
    dy_dt = torch.autograd.grad(
        y_pred,
        t_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # Perte de l'équation différentielle
    loss_DE = criterion(dy_dt + k * y_pred, torch.zeros_like(dy_dt))

    # Perte de la condition initiale : y(0) = 1
    loss_IC = initial_condition_loss(
        model(torch.tensor([[0.0]])), torch.tensor([[1.0]])
    )

    # Perte totale
    loss = loss_DE + loss_IC

    # Rétropropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Affichage de la perte
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
