import numpy as np
from emus import iter_avar


class TestBuildFixedDerivMats():
    def test_stochastic_H(self):
        L = 8
        kappa = np.random.rand(L) + 1e-3
        kappa /= np.sum(kappa)

        nbrs = [np.arange(i, 5+i) % L for i in range(L)]
        psis = np.random.rand(L, 100, 5) + np.random.rand(L, 1, 5)
        psi_sum = np.sum(psis, axis=2, keepdims=True)
        psis /= psi_sum
        g1 = np.random.randn(L, 100)
        g2 = np.random.randn(L, 100)

        H, beta = iter_avar._build_fixed_deriv_mats(psis, g1, g2, nbrs, kappa)
        Hsum = np.sum(H, axis=1)
        assert(np.allclose(Hsum, np.zeros(L)))


class TestBuildNormedTrajs():
    def test_stochastic(self):
        # Build Data
        L = 8
        nbrs = [np.arange(i, 5+i) % L for i in range(L)]
        kappa = np.random.rand(L)
        kappa /= np.sum(kappa)
        state_fe = np.random.randn(L)

        psis = np.random.rand(L, 100, 5)
        g1 = np.random.randn(L, 100)
        g2 = np.random.randn(L, 100)

        # Test Stochastic
        out = iter_avar._build_normed_trajs(psis, state_fe, g1, g2, nbrs, kappa)
        normed_psis = out[0]

        for psis_i in normed_psis:
            normed_psi = np.sum(psis_i, axis=1)
            assert(np.allclose(normed_psi, np.ones(len(normed_psi))))

    def test_w_and_psi_matches(self):
        # Build Data
        L = 5
        nbrs = [np.arange(L) for i in range(L)]
        kappa = np.random.rand(L)
        kappa /= np.sum(kappa)
        state_fe = np.random.randn(L)

        psis = np.random.rand(L, 100, 5)
        g1 = psis[:, :, 0]
        g2 = psis[:, :, 1]

        # Test Stochastic
        out = iter_avar._build_normed_trajs(psis, state_fe, g1, g2, nbrs, kappa)
        normed_psis, normed_w1, normed_w2 = out

        for i in range(L):
            np_i = normed_psis[i]
            nw1_i = normed_w1[i] * kappa[0] / np.exp(-state_fe[0])
            nw2_i = normed_w2[i] * kappa[1] / np.exp(-state_fe[1])
            assert(np.allclose(nw1_i, np_i[:, 0]))
            assert(np.allclose(nw2_i, np_i[:, 1]))


class TestBuildErrTrajs():
    def test_check_psi_sum(self):
        L = 6
        nbrs = [np.arange(i-1, 2+i) % L for i in range(L)]
        psis = np.random.rand(L, 100, 3)
        g1 = np.random.randn(L, 100)
        g2 = np.random.randn(L, 100)

        partial_derivs = np.zeros(L+2)
        partial_derivs[:L] += 1
        kappa = np.ones(L)
        err_trajs = iter_avar._build_err_trajs(psis, g1, g2, partial_derivs, kappa, nbrs)
        for j in range(L):
            xi_j = err_trajs[j]
            assert(np.allclose(xi_j, np.sum(psis[j], axis=1)))

    def test_check_w_sum(self):
        L = 6
        nbrs = [np.arange(i-1, 2+i) % L for i in range(L)]
        psis = np.random.rand(L, 100, 3)
        g1 = np.random.randn(L, 100)
        g2 = np.random.randn(L, 100)

        true_xi = g1 + 2 * g2
        partial_derivs = np.zeros(L+2)
        partial_derivs[-2:] += np.array([1, 2])

        kappa = np.ones(L)
        err_trajs = iter_avar._build_err_trajs(psis, g1, g2, partial_derivs, kappa, nbrs)

        for j in range(L):
            xi_j = err_trajs[j]
            assert(np.allclose(xi_j, true_xi[j]))
        # raise NotImplementedError


class TestGetIatsAndAcovFromTraj():
    def test_given_iats(self):
        L = 8
        N = 10000
        np.random.seed(8675309)
        trajs = np.random.randn(L, N)
        iat_method = np.random.rand(L)
        iat_method /= np.sum(iat_method)

        acov, contribs, iats = iter_avar._get_iats_and_acov_from_traj(trajs, iat_method)
        assert(np.allclose(iat_method, iat_method))
        assert(np.abs(np.sum(contribs * N) - 1) < 0.1)

    def test_calculate_iats(self):
        L = 5
        freq = 10
        N = 10000 * freq
        np.random.seed(8675309)
        trajs = [np.random.randn(L, 1) * np.ones(freq) for i in range(N // freq)]
        trajs = np.hstack(trajs)
        iat_method = 'acor'

        iats = iter_avar._get_iats_and_acov_from_traj(trajs, iat_method)[2]
        assert(np.linalg.norm(iats - freq) / np.sqrt(L) < 1)
