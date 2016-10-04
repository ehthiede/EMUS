import numpy as np

for i in xrange(20):
    data = np.loadtxt('umbrella_%d.txt'%i)
    traj = data[:,1:]
    fdata = np.zeros((len(data),3))
    fdata[:,1] = np.array([(traj>-120) & (traj<-70)]).flatten()
    fdata[:,2] = np.array([(traj>25) & (traj<100)]).flatten()
    fdata[:,0] = data[:,0]
    np.savetxt('fdata_%d.txt'%i,fdata)
    print r'data/fdata_%d.txt'%i


