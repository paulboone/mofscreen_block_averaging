
# methodology from
# (1) Flyvbjerg, H.; Petersen, H. G. Error Estimates on Averages of Correlated Data.
# The Journal of Chemical Physics 1989, 91 (1), 461–466.

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


filename = "lammpstrj.npy"

# row:molecule:x,y,z
data = np.load(filename)
num_rows, num_molecules, num_cols = data.shape
d0 = np.diff(data[:,0,0]) # delta x for first molecule

# max_row = 4096
max_row = 4096000
num_compressions = int(math.log(max_row, 2))
df = pd.DataFrame(d0[0:max_row])
mean = df.mean()

block_transformations = np.arange(0,num_compressions)
block_sizes = 2**block_transformations
variances = np.array([df.groupby(lambda x: x // i).mean().var(ddof=0) for i in block_sizes])
total_points = max_row / block_sizes
adj_variances = variances / ((total_points-1)[:,None])
std_dev = np.sqrt(adj_variances)
errors = (1/np.sqrt(2*total_points)) * std_dev[:,0] # error for stddev

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(block_transformations,std_dev, errors, fmt="o", zorder=2)
ax.set_xlabel('# block transformations applied')
ax.set_xticks(np.arange(num_compressions))
ax.set_ylabel('stddev')
ax.grid(linestyle='-', color='0.7', zorder=0)

fig.savefig("delta_x_corr_time.png", dpi=144)
