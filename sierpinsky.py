import torch
import matplotlib.pyplot as plt
import time 

start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Vertices of equilateral triangle
vertices = torch.tensor([[0, 0], [1, 0], [0.5, 0.866]], device=device)

# Number of points to plot
N = 1_000_000
# generate N random (x, y) points
points = torch.rand((N, 2), device=device)

# Iterate chaos game
steps = 200
for _ in range(steps):
    # generate random value of 0-2, then convert to vertex equivalent
    random_vertex = vertices[torch.randint(0, 3, (N,), device=device)]
    points = (points + random_vertex) / 2

# convert tensor back to numpy array
points_cpu = points.cpu().numpy()
print(time.time() - start_time)
# plot x, y coordinates -> select all rows and respective x, y
plt.scatter(points_cpu[:,0], points_cpu[:,1], s=0.1, color="black")
plt.axis("equal"); # make everything more square
plt.show()