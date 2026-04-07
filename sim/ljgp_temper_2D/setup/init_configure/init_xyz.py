#!/usr/bin/env python3

import numpy as np

N = 1024
rho = 0.72
L = np.sqrt(N / rho)

nx = ny = int(np.sqrt(N))
dx = L / nx

with open("lj2d.data", "w") as f:
    f.write("2D Lennard-Jones system\n\n")
    f.write(f"{N} atoms\n")
    f.write("1 atom types\n\n")
    f.write(f"0.0 {L} xlo xhi\n")
    f.write(f"0.0 {L} ylo yhi\n")
    f.write("-0.5 0.5 zlo zhi\n\n")

    f.write("Masses\n\n1 1.0\n\nAtoms\n\n")

    atom_id = 1
    for i in range(nx):
        for j in range(ny):
            x = i * dx
            y = j * dx
            f.write(f"{atom_id} 1 {x} {y} 0.0\n")
            atom_id += 1
