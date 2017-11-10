import pytest
import numpy as np

def read_mmn(file_mmn,size,klist1d,klist3d,transpose=False):

    with open(file_mmn,'r') as mmn:
        next(mmn)
        nb,nk,nn = map(int,next(mmn).split())


        overlaps = np.zeros((nk,3,2,nb,nb),complex)
        assert(nk == len(klist3d.flatten()) )

        for i in range(nk):
            for j in range(nn):
                k1,k2,g1,g2,g3 = map(int,mmn.next().split())
                assert(i+1 == k1)

                i1,i2,i3 = map(int,np.rint(size*klist1d[i]))

                if k2 == klist3d[(i1+1)%size[0],i2,i3]+1:
                    direction = 0
                    step_size = 0
                    read_block = True
                elif k2 == klist3d[i1,(i2+1)%size[1],i3]+1:
                    direction = 1
                    step_size = 0
                    read_block = True
                elif k2 == klist3d[i1,i2,(i3+1)%size[2]]+1:
                    direction = 2
                    step_size = 0
                    read_block = True
                elif k2 == klist3d[(i1-1)%size[0],i2,i3]+1:
                    direction = 0
                    step_size = -1
                    read_block = True
                elif k2 == klist3d[i1,(i2-1)%size[1],i3]+1:
                    direction = 1
                    step_size = -1
                    read_block = True
                elif k2 == klist3d[i1,i2,(i3-1)%size[2]]+1:
                    direction = 2
                    step_size = -1
                    read_block = True
                else:
                    read_block = False

                if read_block:
                    for k in range(nb*nb):
                        row,column = k//nb, k%nb
                        line = mmn.next()
                        value = sum(np.array(map(float,line.split()))*np.array([1.0,1j]))
                        overlaps[i,direction,step_size,row,column] = value
                    if transpose == True:
                        overlaps[i,direction,step_size,:,:] = overlaps[i,direction,step_size,:,:].T
                else:
                    for k in range(nb*nb):
                        mmn.next()
    return overlaps
