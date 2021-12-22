import numpy as np
from emus import iter_avar_new as iter_avar

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
