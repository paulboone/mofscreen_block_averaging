
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


filename = "lammpstrj.npy"

# row:molecule:x,y,z
data = np.load(filename)
num_rows, num_molecules, num_cols = data.shape


d1 = data[:,1,:]
d2 = data[:,2,:]

#create dataframe with 1024 rows
df = pd.DataFrame(d1[0:1024])

mean = df.mean()
variances = np.array([df.groupby(lambda x: x // (2**i)).mean().var(ddof=0) for i in range(0,9)])
total_points = np.array([1024 / 2**i for i in range(0,9)])
adj_variances = variances / ((total_points-1)[:,None])

errors = np.sqrt(2/points_per_group) * adj_variances[:,1]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(range(0,9),adj_variances[:,1], errors, zorder=2)
ax.set_xlabel('# block transformations applied')
ax.set_ylabel('variance / (N-1)')
