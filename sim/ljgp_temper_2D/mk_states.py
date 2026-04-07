#!/usr/bin/env python3

#################################################################################################################################
# Make a list of states (epsilon, r0) for LJGP MD simulations.                                                                  #
# Follow "Self-Assembly of Monatomic Complex Crystals and Quasicrystals with a Double-Well Interaction Potential"
# Parameter eplison is in a set of (0, 5] with an interval step of 0.1                                                          #
# Parameter r0 is in a set of (1, 2.1] with a step of 0.01                                                                      #
# Total number of states = 5500                                                                                                 #
#################################################################################################################################

import numpy as np

count = 0
for eps_i in range(50):
    for r0_i in range(100):
        epsilon = (eps_i + 1) * 0.1
        r0 = (r0_i + 1 ) *0.01 + 1.1

        print(f"{count:04d} {epsilon: .4f} {r0: .4f}")
        count += 1
