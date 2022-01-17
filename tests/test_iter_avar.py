# -*- coding: utf-8 -*-
from __future__ import division

import pytest
import numpy as np

from emus import iter_avar


# @pytest.mark.parametrize('iat_method', ['acor', 'ipce', 5.])
# def test_get_repexchange_avars(random_plateaus, iat_method):
#     true_tau = 5.
#     iat_tol = 0.3
#     var_tol = 1e-4
#     test_data = random_plateaus
#     true_acovar = 1. * (true_tau / len(test_data))
#     iat, acovar = iter_avar._get_repexchange_avars(test_data, iat_method)
#     assert(np.abs(iat - true_tau) < iat_tol)
#     assert(np.abs(acovar - true_acovar) < var_tol)


# @pytest.mark.parametrize('iat_method', ['acor', 'ipce', 5.])
# def test_get_iid_avars(random_plateaus, iat_method):
#     Nstates = 10
#     if type(iat_method) is not str:
#         print('inside', iat_method)
#         iat_method = np.array([iat_method] * Nstates)
#     true_tau = 5.
#     iat_tol = 0.3
#     var_tol = 1e-4
#     test_data = random_plateaus
#     true_acovar = 1. * (true_tau / len(test_data))
#     test_data = [random_plateaus] * Nstates
#     iat, acovar = iter_avar._get_iid_avars(test_data, iat_method)
#     assert(np.allclose(iat, true_tau, atol=iat_tol, rtol=0.))
#     assert(np.allclose(acovar, true_acovar, atol=var_tol, rtol=0.))
