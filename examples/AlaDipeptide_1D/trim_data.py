import numpy as np

L = 20
for i in xrange(20):
    traj = np.loadtxt('data/trajw_um%d_trim5k.txt'%i)
    traj = traj[::10]
    print i, np.shape(traj)
    np.savetxt('data/umbrella_%d.txt'%i,traj)

