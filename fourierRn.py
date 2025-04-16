import torch
import torch.nn as nn
import numpy as np

# FourierNet Definition for Periodic PDEs
class FourierNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, output_size=1):
        super(FourierNet, self).__init__()
        self.hidden_size = hidden_size

        # Frequency embedding with sinusoids
        self.freqs = nn.Parameter(torch.linspace(1.0, 10.0, hidden_size).reshape(1, -1))
        self.linear_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x_proj = x @ self.freqs  # projection: [batch, hidden_size]
        sin_feat = torch.sin(2 * np.pi * x_proj)
        cos_feat = torch.cos(2 * np.pi * x_proj)
        fourier_feat = torch.cat([sin_feat, cos_feat], dim=-1)
        out = self.linear_out(fourier_feat)
        return out


#  Compute derivatives

def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def laplacian(y, x):
    grad_y = gradient(y, x)
    return gradient(grad_y, x)


def run_solver(problem='heat'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FourierNet(input_size=2, hidden_size=50).to(device)

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    xt = np.stack([X.flatten(), T.flatten()], axis=1).astype(np.float32)
    xt_tensor = torch.tensor(xt, requires_grad=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    for epoch in range(5000):
        u = model(xt_tensor)

        x_input = xt_tensor[:, 0:1].clone().detach().requires_grad_(True)
        t_input = xt_tensor[:, 1:2].clone().detach().requires_grad_(True)
        xt_input = torch.cat([x_input, t_input], dim=1).to(device)
        u = model(xt_input)

        u_t = gradient(u, t_input)
        u_x = gradient(u, x_input)
        u_xx = gradient(u_x, x_input)
        u_tt = gradient(u_t, t_input)

        if problem == 'heat':
            alpha = 0.1
            pde_residual = u_t - alpha * u_xx
        elif problem == 'wave':
            c = 1.0
            pde_residual = u_tt - c ** 2 * u_xx
        elif problem == 'schrodinger':
            u_real, u_imag = torch.chunk(u, 2, dim=1)
            u = u_real + 1j * u_imag
            u_xx = laplacian(u, x_input)
            u_t = gradient(u, t_input)
            pde_residual = u_t + 0.5j * u_xx
            loss = torch.mean(torch.abs(pde_residual) ** 2)
        elif problem == 'laplace':
            pde_residual = u_xx
        elif problem == 'poisson':
            rho = torch.sin(np.pi * x_input).to(device)
            pde_residual = u_xx + rho
        elif problem == 'pendulum':
            u_tt = gradient(gradient(u, t_input), t_input)
            pde_residual = u_tt + torch.sin(u)
        else:
            raise ValueError("Unsupported problem type")

        if problem not in ['schrodinger']:
            loss = mse(pde_residual, torch.zeros_like(pde_residual))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


# Run for different PDEs:

run_solver(problem='heat')        # Equation de la chaleur
# run_solver(problem='wave')      #  Equation d'onde
# run_solver(problem='schrodinger') # Equation de Schrödinger (complex)
# run_solver(problem='laplace')   # Equation de Laplace
# run_solver(problem='poisson')   # Equation de Poisson
# run_solver(problem='pendulum')  #  EDO périodique non linéaire
