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
num_molecules = 10
num_cols = 3 # x, y, z (for COM)
data = np.zeros((num_rows, num_molecules, num_cols))
start_molecule = 1

with open(filename, 'r') as f:
    for row in range(num_rows):
        _ = next(f) # timestep
        timestep = int(next(f))

        _ = next(f) # number of atoms
        num_atoms = int(next(f))

        _ = next(f) # box bounds
        _ = next(f) # box bounds x
        _ = next(f) # box bounds y
        _ = next(f) # box bounds z

        _ = next(f) # atoms


        masses = np.zeros(num_molecules)

        for a in range(num_atoms):
            cols = next(f).strip().split()
            mol_index = int(cols[2]) - start_molecule
            mass = float(cols[6])
            x = float(cols[8])
            y = float(cols[9])
            z = float(cols[10])

            masses[mol_index] += mass
            data[row, mol_index, 0] += x * mass
            data[row, mol_index, 1] += y * mass
            data[row, mol_index, 2] += z * mass

        # divide x,y,z by total mass
        for mol_index in range(num_molecules):
            data[row, mol_index] /= masses[mol_index]

np.save('lammpstrj.npy', data)



# try:
#     for filename in args.filenames:
# except BrokenPipeError:
#     print("Broken Pipe. Stdout likely closed by program output is piped to.")
