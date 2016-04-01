import numpy as np
import usroutines as usr
from emus import emus
import usroutines as usr
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

#psis = np.load('testpsis.npy')
def main():
    L = 20
    nbrlist = np.array([(np.arange(5)-2+i)%L for i in range(L)])
#    nbrlist = np.array([(np.arange(3)-1+i)%L for i in range(L)])
    print nbrlist
    psis = np.load('q2.npy')
    psinbr = []
    for i in xrange(L):
        psi_j =[]
        for j in nbrlist[i]:
            psi_j.append(psis[i,:,j])
        psinbr.append(psi_j)
    psinbr = np.array(psinbr)
    psinbr = psinbr.transpose(0,2,1)
    EM = emus(psis)
    z = EM.calc_zs()
#    z_MB1 = EM.calc_zs(npolish=1,usetaus=False)
    z_MB2 = EM.calc_zs(npolish=10,usetaus=False)
#    z_MB3 = EM.calc_zs(npolish=20,usetaus=True)
    EM2 = emus(psinbr,neighborlist=nbrlist)
    print dir(EM2)
    print EM2.neighborlist
    z2 = EM2.calc_zs()
    z_MB3 = EM2.calc_zs(npolish=10,usetaus=False)
    z_MB4 = EM2.calc_zs(npolish=10,usetaus=True)

    #zref, Fref = usr.emus_iter(psis)

    print "##################"
    #print np.max(np.abs(z - zref) / zref)
    print "##################"
    #np.save("z_emustest",z)
    #np.save("z_MB1",z_MB1)
    #np.save("z_MB2",z_MB2)

    plt.plot(-np.log(z), label="Emus")
    plt.plot(-np.log(z2), label="Emus2")
#    plt.plot(-np.log(z_MB1), label="MB1")
    plt.plot(-np.log(z_MB2), label="MB2")
    plt.plot(-np.log(z_MB3), label="MB3",marker='.')
    plt.plot(-np.log(z_MB4), label="MB4",marker='.')

    refdir = "/home/thiede/scratch-midway/1DGMXPL_NOCMAP_LONG/run0/data/"
    emus_z_ref = -np.log(np.loadtxt(refdir+'emusz_trim5k.txt'))
    emus_F_ref = -np.log(np.loadtxt(refdir+'emusF_trim5k.txt'))

    #print np.sum(np.abs(emus_F_ref-Fref)/emus_F_ref)
    #print np.max(np.abs(emus_z_ref - zref)/zref)
    print "MAXES!"
    #print np.max(Fref), np.min(Fref)
    #print np.max(emus_F_ref), np.min(emus_F_ref)
    print "MAXES"
    print np.shape(emus_F_ref)
    #print np.abs(Fref - emus_F_ref)
    MB1_z_ref = -np.log(np.loadtxt(refdir+'MB1z_trim5k_taus.txt'))
    MB2_z_ref = -np.log(np.loadtxt(refdir+'MB2z_trim5k_taus.txt'))
    #plt.plot(emus_z_ref,label='EMref')
    #plt.plot(MB1_z_ref,label='MB1ref')
    #plt.plot(MB2_z_ref,label='MB2ref')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
