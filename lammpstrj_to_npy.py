#!/usr/bin/env python3

import argparse
# import csv
import sys

import numpy as np
# from lammps_tools.utils import thermo_from_lammps_log

# parser = argparse.ArgumentParser("./lammpstrj2np.py")
# parser.add_argument('filenames', nargs='+', help="Path(s) to LAMMPS average output file(s)")
# args = parser.parse_args()
#
# cols = []
# last_timestep = -1
# tsv = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
#

filename = "gas.lammpstrj"

num_rows = 1985
num_atoms = 30
num_cols = 7 # id, type, mol, mass, x, y, z
data = np.zeros((num_rows, num_atoms, num_cols))

with open(filename, 'r') as f:
    for i in range(num_rows):
        _ = next(f) # timestep
        timestep = int(next(f))

        _ = next(f) # number of atoms
        num_atoms = int(next(f))

        _ = next(f) # box bounds
        _ = next(f) # box bounds x
        _ = next(f) # box bounds y
        _ = next(f) # box bounds z

        _ = next(f) # atoms

        for a in range(num_atoms):
            cols = next(f).strip().split()
            cols = cols[0:3] + cols[6:7] + cols[8:11]
            data[i,a] = [float(s) for s in cols]


np.save('lammpstrj.npy', data)



# try:
#     for filename in args.filenames:
# except BrokenPipeError:
#     print("Broken Pipe. Stdout likely closed by program output is piped to.")
