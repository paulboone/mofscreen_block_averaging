
# methodology from
# (1) Flyvbjerg, H.; Petersen, H. G. Error Estimates on Averages of Correlated Data.
# The Journal of Chemical Physics 1989, 91 (1), 461–466.

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


filename = "lammpstrj.npy"

# row:molecule:x,y,z
data = np.load(filename)
num_rows, num_molecules, num_cols = data.shape
d0 = data[:,0,0] # x for first molecule

# max_row = 4096
max_row = 50000
corr_time = 2048
num_molecules = 10

start_time = time.time()
prev_time = start_time

norm = np.zeros(max_row)
sumd = np.zeros((max_row, 3))
origin = np.zeros((math.ceil(max_row / corr_time), num_molecules, 3))
for row_index in range(max_row):
    rel_index = row_index % corr_time

    if rel_index == 0:
        origin[row_index // corr_time] = data[row_index]

    for origin_index in range(row_index // corr_time + 1):
        drow = row_index - origin_index * corr_time
        sumd[drow] += (data[row_index] - origin[origin_index]).sum(axis=0)**2
        norm[drow] += num_molecules

    # print out progress
    if row_index % 10000 == 0:
        t = time.time()
        print("%.1fs (delta %5.1fs): %i" % (t - start_time, t - prev_time, row_index))
        prev_time = t

sumd /= norm[:,None]

np.save('tau_vs_t.npy', sumd)
np.savetxt('tau_vs_t.txt', sumd)

#
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1, 1, 1)
# ax.errorbar(block_transformations,std_dev, errors, fmt="o", zorder=2)
# ax.set_xlabel('# block transformations applied')
# ax.set_xticks(np.arange(num_compressions))
# ax.set_ylabel('stddev')
# ax.grid(linestyle='-', color='0.7', zorder=0)
#
# fig.savefig("delta_x_corr_time.png", dpi=144)
