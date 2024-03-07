import deepxde as dde
from deepxde.backend import tf
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        
    def forward(self,x):
        return self.net(x)

def initial_condition(x,y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def boundary_condition(x,y,t, custom_value):
    return torch.full_like(x, custom_value)                         # It is like np.zeros_like but this will create the size of x tensor with custom_value filled in the tensor

def generate_training_data(num_points):                             # Domain points to calculate the PDE loss
    x = torch.rand(num_points, 1, requires_grad= True)              # size = (n, 1) in x direction between 0 to 1
    y = torch.rand(num_points, 1, requires_grad= True)              # size = (n, 1) in y direction between 0 to 1
    t = torch.rand(num_points, 1, requires_grad= True)              # size = (n, 1) for time between 0 to 1
    return x,y,t

def generate_boundary_points(num_points):                           # Boundary Points to calculate the NN loss
    x_boundary = torch.tensor([0.0,1.0]).repeat(num_points//2)      # '//' operator is floor after devide, so 11/2 = 5.
    y_boundary = torch.rand(num_points)                           # This and line above gives the bc for x. The line above only generate points 0 and 1 (boundary points) and this line produce corresponding y points. So u(0,y) and u(1,y)
    
    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary             # Based on the generated random value, we switch the boundary. So u(0/1,y) becomes u(x,0/1)
    
    return x_boundary.view(-1,1), y_boundary.view(-1,1)             # Returning with reshaping so that it is a column vector. Any number of rows but columns are only 1.

def generate_boundary_training_data(num_points):                    # Boundary Points to calculate the NN loss
    x_boundary, y_boundary = generate_boundary_points(num_points)
    t = torch.rand(num_points, 1, requires_grad= True)
    
    return x_boundary, y_boundary, t


def pde(x,y,t,model):
    input_data = torch.cat([x,y,t],dim=1)                           # Inside domain points, size = (n, 3)
    
    u = model(input_data)                                           # u calculated from NN model
    
    u_x, u_y = torch.autograd.grad(                                 # du_dx and du_dy calculating
        u,
        [x,y],
        grad_outputs = torch.ones_like(u),
        create_graph = True,
        retain_graph = True
        )
    
    u_xx = torch.autograd.grad(                                     # d2u_dx2 calculating
        u_x,
        x,
        grad_outputs = torch.ones_like(u_x),
        create_graph = True,
        retain_graph = True
        )[0]
    
    u_yy = torch.autograd.grad(                                     # d2u_dy2 calculating
        u_y,
        y,
        grad_outputs = torch.ones_like(u_y),
        create_graph = True,
        retain_graph = True
        )[0]
    
    u_t = torch.autograd.grad(                                      # du_dt calculating
        u,
        t,
        grad_outputs = torch.ones_like(u),
        create_graph = True,
        retain_graph = True
        )[0]
    
    heat_eq_residual = 1 * u_xx + 1* u_yy - u_t                    # Heat eq, where alpha = 1
    
    return heat_eq_residual
    

def train_PINN(model, num_iterations, num_points):                  # Training loop with the optimizer (Only Adam this time)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        x,y,t = generate_training_data(num_points)                  # Training data inside the domain, for PDE loss
        
        x_b,y_b,t_b = generate_boundary_training_data(num_points)   # Training data on boundary for normal loss from net
        
        t_initial = torch.zeros_like(t)
        u_initial = initial_condition(x,y)                          # Need true u to calculate the loss from the initial and boundary points calculated from net
        
        custom_value = 0
        u_boundary_x = boundary_condition(x_b,y_b,t_b,custom_value) # Need true u at boundary points at given t for the loss from the net
        u_boundary_y = boundary_condition(y_b,x_b,t_b,custom_value) # Need true u at boundary points at given t for the loss from the net
        
        residual = pde(x,y,t,model)
        
        loss = nn.MSELoss()(u_initial, model(torch.cat([x,y,t_initial], dim=1))) + \
                nn.MSELoss()(u_boundary_x, model(torch.cat([x_b,y_b,t_b], dim=1))) + \
                nn.MSELoss()(u_boundary_y, model(torch.cat([y_b,x_b,t_b], dim=1))) + \
                nn.MSELoss()(residual, torch.zeros_like(residual))
        # u_initial = true u at t=0, model() calculate the u from NN for initial condition
        # u_boundary_x = true u at x=0 and 1, model() calculate the u from NN for initial condition
        # u_boundary_y = true u at y=0 and 1, model() calculate the u from NN for initial condition
        # residualo = pde residual (all term on left, so on right = 0 -> torch.zeros_like)
        
        loss.backward()                                             # Back propogation
        
        optimizer.step()                                            # Optimization within that iteration
        
        if iteration % 100 == 0:
            print("iteration", iteration, "loss", loss)

model = PINN()

num_iterations = 10000
num_points = 1000

train_PINN(model, num_iterations, num_points)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('Coursework\Simple_PINN\PINN_2D_Heat.pt') 
#torch.save(model, 'PINN_2D_Heat.pt')