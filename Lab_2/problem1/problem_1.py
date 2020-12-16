import numpy as np
import gym
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import time


# Load model
try:
    model = torch.load('neural-network-avg_141.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)

w = np.linspace(-math.pi, math.pi, 64)
y = np.linspace(0, 1.5, 64)
W, Y = np.meshgrid(w, y)

Z = np.empty(W.shape)
Z_arg = np.empty(W.shape)

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        state = np.array([0, Y[i,j], 0, 0, W[i,j], 0, 0, 0])
        state_tensor = torch.tensor([state],
                                    requires_grad=True,
                                    dtype=torch.float32)
        Q_values = model.forward(state_tensor)
        
        Z[i,j] = Q_values.max(1)[0].item()
        Z_arg[i,j] = Q_values.max(1)[1].item()
        if Q_values.max(1)[1].item() == 2:
            print(Q_values.max(1)[1].item())

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('w')
plt.ylabel('y')
plt.title('Q value')

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, Y, Z_arg, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('w')
plt.ylabel('y')
plt.title('Q action')
plt.show()