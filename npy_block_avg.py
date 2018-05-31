
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


filename = "lammpstrj.npy"

# row:molecule:x,y,z
data = np.load(filename)
num_rows, num_molecules, num_cols = data.shape


d1 = data[:,1,:]
d2 = data[:,2,:]

max_row = 1024
#create dataframe with 1024 rows
df = pd.DataFrame(d1[0:max_row])

mean = df.mean()

block_transformations = np.arange(0,9)
block_sizes = 2**block_transformations
variances = np.array([df.groupby(lambda x: x // i).mean().var(ddof=0) for i in block_sizes])

total_points = max_row / block_sizes
adj_variances = variances / ((total_points-1)[:,None])

errors = np.sqrt(2/total_points) * adj_variances[:,1]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(block_transformations,adj_variances[:,1], errors, zorder=2)
ax.set_xlabel('# block transformations applied')
ax.set_ylabel('variance / (N-1)')
