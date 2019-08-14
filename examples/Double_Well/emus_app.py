import numpy as np
from scipy import integrate
import emus
from emus import emus, avar, iter_avar, linalg
import matplotlib.pyplot as plt

# Define Simulation Parameters
dim = 1                             # 1 Dimensional CV space.


def force_fxn(i, x, L, potential_factor=1):
    """
    Calculates the biased force
    """
    F_unbiased = (-2 * np.pi * np.sin(2 * np.pi * x) - 8 *
                  np.sin(4 * np.pi * x) * np.pi) * potential_factor
    F_biased = -2 * np.pi * L * \
        np.sin(np.pi * (x - i / L)) * np.cos(np.pi * (x - i / L))
    return F_biased + F_unbiased


def unbiased_potential(x, potential_factor=1):
    """
    """
    return (-np.cos(2 * np.pi * x) - 2 *
            np.cos(4 * np.pi * x)) * potential_factor


def bias_potential(i, x, L):
    return L * (np.sin(np.pi * (x - i / L)))**2


def simulate_walker(L, nsteps=1E5, burnin=100, dt=0.001, kT=1.):
    """
    Runs a Brownian motion trajectory on the biased states.
    """
    cv_trajs = []
    for i in range(L):
        cfg_0 = i / L
        cfg = cfg_0
        R_n = np.random.normal(0, 1)
        traj = []
        sig = np.sqrt(kT * dt / 2)
        for j in range(int(nsteps + burnin)):
            rando = np.random.normal(0, 1)
            force = force_fxn(i, cfg, L)
            cfg += dt * force + sig * (rando + R_n)
            R_n = rando
            traj.append(cfg % 1)
        cv_trajs.append(traj[burnin:])
    return cv_trajs


def simulate_iid(L, size=1000, n_control=1E6, nLoop=0):
    nLoop = 0
    size = 1000
    cv_trajs = []
    for i in range(L):
        traj = []
        while len(traj) < size and nLoop < int(n_control):
            x = np.random.uniform(low=0, high=1)
            prop = np.exp(-unbiased_potential(x) -
                          bias_potential(i, x, L)) / 50
            assert prop >= 0 and prop <= 1
            if np.random.uniform(low=0, high=1) <= prop:
                traj += [x]
            nLoop += 1
        cv_trajs.append(np.array(traj) % 1)
    return cv_trajs


def get_psi(a, i, L, kT=1.):
    return np.exp(-bias_potential(i, a, L) / kT)


def z_calc(L, kT=1.):
    z1 = np.array([integrate.quad(lambda x, i=i:get_psi(
        x, i, L, kT=kT) * np.exp(-unbiased_potential(x)), 0, 1)[0] for i in range(L)])
    z1 = z1 / np.sum(z1)
    return z1


def z_sim_iter(cv_trajs, kT=1., neighbors=None):
    L = len(cv_trajs)
    if neighbors is None:
        neighbors = [np.arange(L).astype('int') for i in np.arange(L)]

    psis = []
    for i in range(L):
        psi = np.zeros((len(cv_trajs[i]), L))
        ut = np.zeros(L)
        for t in range(len(cv_trajs[i])):
            ut = np.array([bias_potential(j, cv_trajs[i][t], L)
                           for j in range(L)])
            psi[t] = np.exp(-ut / kT)
        psis.append(psi)
    z_iter, F_iter = emus.calculate_zs(psis, neighbors=neighbors, n_iter=5)
    return z_iter


def z_sim_noniter(cv_trajs, kT=1.):
    L = len(cv_trajs)
    psis = []
    for i in range(L):
        psi = np.zeros((len(cv_trajs[i]), L))
        ut = np.zeros(L)
        for t in range(len(cv_trajs[i])):
            ut = np.array([bias_potential(j, cv_trajs[i][t], L)
                           for j in range(L)])
            psi[t] = np.exp(-ut / kT)
        psis.append(psi)
    z, F = emus.calculate_zs(psis, n_iter=0)
    return z, F


def calc_F(x, i, l, m, L, kT, z):
    '''Returns the function \psi_l*\psi_m/ \sum_k(\psi_k/z_k)^2
    '''
    return get_psi(x, l, L, kT) * get_psi(x, m, L, kT) * pi_i(x, i, L) / sum(f(x) for f in [lambda x, k=k:get_psi(x, k, L, kT) / z[k] for k in range(L)])**2


def pi_i(x, i, L):
    r'''Returns the function \pi_i'''
    return np.exp(-unbiased_potential(x) -
                  bias_potential(i, x, L))


def B_calc(z, kT=1):
    ''' Calculate the B matrix by direct integration
    '''
    L = len(z)
    B = np.zeros((L, L))
    for l in range(L):
        for m in range(L):
            # print([scipy.integrate.quad(F(i,l,m),0,1)[0] for i in range(L)])
            # print(np.array([scipy.integrate.quad(F1(i),0,1)[0] for i in range(L)]))
            B[l, m] = -np.sum(np.array([integrate.quad(calc_F, 0, 1, args=(i, l, m, L, kT, z))[0] / integrate.quad(pi_i, 0, 1, args=(i, L))[0] / (z[l]**2) for i in range(L)]))
            if l == m:
                B[l, m] += 1
    return B


def B_sim(cv_trajs, z, L, kT=1.):
    ''' Calculate the B matrix by integration using sample points
    '''
    B = np.zeros((L, L))
    for l in range(L):
        for m in range(L):
            for i in range(L):
                cv_trajs_i = np.array(sorted(np.array(cv_trajs[i])))
                fx = get_psi(cv_trajs_i, l, L, kT) * get_psi(cv_trajs_i, m, L, kT) / np.sum([get_psi(cv_trajs_i, k, L, kT) / z[k] for k in range(L)], axis=0)**2
                pix = np.exp((-unbiased_potential(cv_trajs_i) -
                              bias_potential(i, cv_trajs_i, L)) / kT)
                B[l, m] -= integrate.trapz(np.array(fx * pix), cv_trajs_i) / (integrate.trapz(np.array(pix), cv_trajs_i) * z[m]**2)
    for r in range(L):
        B[r, r] += 1
    return(B)


def calc_a(psis, z):
    L = len(z)
    N = np.zeros(L)
    for i in range(L):
        N[i] = psis[i].shape[0]
    a = []
    for i in range(L):
        ai = np.zeros((int(N[i]), L))
        for t in range(int(N[i])):
            for j in range(L):
                ai[t, j] = (psis[i][t, j] / z[i]) / np.sum(psis[i][t] / z)
        a.append(ai)
    return(a)


def z_error_sim_iter(z, psis):
    zerr_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar.calc_log_z(
        psis, z, repexchange=False)
    return(zerr_iter)


def z_error_sim_noniter(psis, z, F):
    z_err, log_zcontribs, log_ztaus = avar.calc_partition_functions(
        psis, z, F, repexchange=False)
    return(z_err)


def z_error_calc_iter(a, B, z):
    L = B.shape[0]
    B_pseudo_inv = linalg.groupInverse(B)
    zeta_traj = []
    for i in range(L):
        Ni = int(a[i].shape[0])
        zeta_i = np.zeros((Ni, L))
        for t in range(int(Ni)):
            for r in range(L):
                zeta_i[t, r] = z[i] * np.dot(a[i][t], B_pseudo_inv.T[r])
        zeta_traj.append(zeta_i)
    z_contribs = np.zeros((L, L))
    z_iats = np.zeros((L, L))
    for k in range(L):
        # Extract contributions to window k FE.
        zeta_k = [zt[:, k] for zt in zeta_traj]
        z_iats[k], z_contribs[k] = iter_avar._get_iid_avars(
            zeta_k, iat_method='ipce')
    zerr_iter = np.sum(z_contribs, axis=1)
    return zerr_iter


def main():
    kT = 1
    L = 5
    interval = np.arange(L)
    neighbors = [interval.astype('int') for i in interval]

    cv_trajs = simulate_walker(L, kT=kT)
    z1 = z_sim_iter(cv_trajs, kT=kT, neighbors=neighbors)
    z2, F = z_sim_noniter(cv_trajs, kT=kT)
    psis = []
    for i in range(L):
        psi = np.zeros((len(cv_trajs[i]), L))
        ut = np.zeros(L)
        for t in range(len(cv_trajs[i])):
            ut = np.array([bias_potential(j, cv_trajs[i][t], L) for j in range(L)])
            psi[t] = np.exp(-ut / kT)
        psis.append(psi)
    zerr_iter_algorithm = z_error_sim_iter(z1, psis)
    zerr_noniter = z_error_sim_noniter(psis, z2, F)

    '''
    z_list=[z_sim_iter(simulate_walker()) for i in range(100)]
    np.save('z_list.npy',z_list)

    z_list=np.load('z_list.npy')
    z_list=np.array(z_list)
    true_error=np.array([np.var(z_list[:,i]) for i in range(L)])
    '''
    z = z_calc(L, kT=kT)
    B = B_calc(z, kT=kT)
    # B1=B_sim(L, kT=kT)
    B_ref = np.load("B_ref.npy")
    a = calc_a(psis, z)
    zerr_iter = z_error_calc_iter(a, B, z)
    # zerr_iter_2=z_error_calc_iter(B1)

    print("Calculated std in iter_z: ", np.sqrt(zerr_iter / z))
    # print("Calculated using smp points:", np.sqrt(zerr_iter_2/z))
    # print("true error",np.sqrt(true_error/z))
    print("Numerical Result Using iter_avar", np.sqrt(zerr_iter_algorithm / z1))

    # print("Ref Calculated std in iter_z: ",np.sqrt(zerr_iter_1/z_iter))

    plt.plot(zerr_iter / z, label='Analytical Result for Estimated Iter_AVAR', c='b')
    # plt.plot(zerr_iter_2/z, label='Estimated AVAR using smp',c='g')
    # plt.plot(true_error/z, label='True Iter_AVAR',c='g')
    plt.plot(zerr_iter_algorithm / z1, label='Numerical Result for Estimated Iter_AVAR', c='r')
    # plt.plot(zerr_iter_1/z_iter, label='Estimated AVAR from simulation',c='g')
    # plt.plot(zerr_noniter/z2, label='Old AVAR estimate')
    plt.ylabel('Variance in $z$')
    plt.xlabel('Window Index')
    plt.yscale('log')
    plt.legend()
    plt.show()
    '''
    xax = np.arange(0,1,0.01)
    U = unbiased_potential(xax)
    U-= np.min(U)
    fe, hist = emus.calculate_pmf(cv_trajs, psis, (0, 1), z, kT=kT)
    fe -= np.min(fe)
    fe_iter, hist = emus.calculate_pmf(cv_trajs, psis, (0,1), z, kT=kT)
    fe_iter -= np.min(fe_iter)

    plt.plot(xax, U, label="Unbiased Potential",c='r')
    plt.plot(xax, fe, label="fe",c="g")
    plt.plot(xax, fe_iter, "b--", label="fe iter")
    plt.legend()
    plt.show()


    plt.plot(xax, fe, label="fe",c="g")
    plt.show()
    '''


if __name__ == "__main__":
    main()
