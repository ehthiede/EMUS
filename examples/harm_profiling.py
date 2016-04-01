import numpy as np
import usroutines as usr
import time

forceConstant = 0.00760535
fKs = np.ones((144,2))*forceConstant
print np.shape(fKs)
kTs = np.ones(144)
sidelength = 12
hw = 360./sidelength
centersx = np.linspace(-180.0+hw/2,180.0-hw/2,sidelength) 
centersy = np.linspace(-180.0+hw/2,180.0-hw/2,sidelength) 
centers = np.array([[centersx[i%sidelength],centersy[int(i/sidelength)]] for i in xrange(sidelength**2)])

traj = np.load('wassup.npy')
traj = traj.transpose(1,0,2)

st = time.time()
print traj.shape
print "Starting"
psi_i = []
for i, traj_i in enumerate(traj):
    psi_i.append(usr.calc_harmonic_psis(traj_i,centers,fKs,kTs))

print np.shape(psi_i)
print time.time() - st
    
