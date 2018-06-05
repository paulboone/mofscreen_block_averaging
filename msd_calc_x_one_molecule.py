
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
max_row = 4096000
corr_time = 2048

start_time = time.time()
prev_time = start_time


sumdx = np.zeros(max_row)
originx = np.zeros(max_row // corr_time)
for row in range(max_row):
    rel_index = row % corr_time

    for i in range(row // corr_time + 1):
        origin_row = i * corr_time
        drow = row - origin_row
        dx = d0[row] - originx[i]
        sumdx[drow] += dx**2

    if rel_index == 0:
        originx[row // corr_time] = d0[row]


    if row % 10000 == 0:
        t = time.time()
        print("%.1fs (delta %5.1fs): %i" % (t - start_time, t - prev_time, row))
        prev_time = t


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
