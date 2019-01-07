import os
import sys
import h5py
import numpy as np


def nearest_neighbor_table(klist3d, nn):
    """ Generate table of nn nearest neighbors. """
    nk = len(klist3d.flatten())
    nn_table = np.zeros((nk, 3, 2 * nn), int)
    size = klist3d.shape
    for ix in range(size[0]):
        for iy in range(size[1]):
            for iz in range(size[2]):
                i = klist3d[ix, iy, iz]
                for j in range(-nn, nn):
                    if j < 0:
                        nn_table[i, 0, j] = klist3d[(ix + j) % size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy + j) % size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz + j) % size[2]]
                    if j >= 0:
                        # DOES j=0 CORRESPOND TO THE FIRST NEIGHBOR? (VLAD)
                        nn_table[i, 0, j] = klist3d[(ix + j + 1) % size[0], iy, iz]
                        nn_table[i, 1, j] = klist3d[ix, (iy + j + 1) % size[1], iz]
                        nn_table[i, 2, j] = klist3d[ix, iy, (iz + j + 1) % size[2]]
    return nn_table


