import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

g = np.loadtxt("g.txt", delimiter=",")
lamb = np.loadtxt("lambda.txt", delimiter=",")

fig, axes = plt.subplots(1,2, figsize=(10,6))
ax = axes[0]
for i in range(1,10):
    ax.plot(lamb[:,0], lamb[:,i], label=i)
    ax.legend()
    ax.set_title("Lambda (tensor traces)")

ax = axes[1]
for i in range(1,10):
    ax.plot(g[:,0], g[:,i], label=i)
    ax.legend()
    ax.set_title("g coefficients")

plt.tight_layout()
plt.show()

