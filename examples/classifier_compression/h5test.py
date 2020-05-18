import h5py
import numpy


bayesIters = 3
f = h5py.File("mydata.hdf", "w")
dset = f.create_dataset("ostats", shape=(3, 2))
dset[0, 0] = 1
dset[1, 0] = 2
dset[2, 0] = 3
f.close()

r = h5py.File("mydata.hdf", "r")
print(list(r.keys()))
dr = r['ostats']
print(dr[0,1])