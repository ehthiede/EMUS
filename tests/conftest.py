# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest


@pytest.fixture(scope='session')
def random_plateaus():
    np.random.seed(8675309)
    nrep = 5
    N = 1000
    x = np.random.randn(N)
    X = np.stack([x for i in range(nrep)]).T
    X = X.ravel()
    return X
