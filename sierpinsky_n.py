import torch
import matplotlib.pyplot as plt
import time 
import numpy as np

start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Sides of shape
SIDES = 6

# create N point unit circle
r = np.arange(SIDES) # create evenly spaced array size N
points = np.exp(2.0 * np.pi * 1j * r/ SIDES)
# rotate clockwise to make sides straight
points = points * np.exp(1j * np.angle(points[0]-points[1]))
vertices = torch.tensor(np.column_stack((points.real, points.imag)), device=device)

# Number of points to plot
N = 1_000_000
# generate N random (x, y) points
points = torch.rand((N, 2), device=device)

# Iterate chaos game
steps = 200
for _ in range(steps):
    # generate random value of 0-2, then convert to vertex equivalent
    random_vertex = vertices[torch.randint(0, SIDES, (N,), device=device)]
    points = (points + random_vertex) / (SIDES - 1)

# convert tensor back to numpy array
points_cpu = points.cpu().numpy()
print(time.time() - start_time)
# plot x, y coordinates -> select all rows and respective x, y
plt.scatter(points_cpu[:,0], points_cpu[:,1], s=0.1, color="black")
plt.axis("equal"); # make everything more square
plt.show()