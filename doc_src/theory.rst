Background
==========

Here, we give a quick overview of the theory behind the EMUS package.  For an in-depth analysis, please refer to the [EMUS paper](http://scitation.aip.org/content/aip/journal/jcp/145/8/10.1063/1.4960649).

Basics of Equilibrium Umbrella Sampling
---------------------------------------

In umbrella sampling, we are interested in computing averages of a function :math:`g` over a probability distribution :math:`\pi`.  We will use angle brackets to denote the the value of these averages (also known as expectations).

.. math:: \left \langle g \right \rangle = \int g(x) \pi(x) dx.

For instance, :math:`\pi` might be the Boltzmann distribution,

.. math:: \pi(x) = \frac{\exp \left(- H_0(x) / k_B T \right)}{\int \exp \left(- H_0(x)/ k_B T \right) dx}

where :math:`H_0` is the system Hamiltonian,  :math:`k_B` is Boltzmann's constant, :math:`T` is the temperature.  In particular, we can express the free energy difference between two states :math:`S_1` and :math:`S_2` as

.. math:: \Delta G = -k_B T \ln \left( \frac{\left \langle \iota_{S1} \right \rangle}{ \left \langle \iota_{S2} \right \rangle}\right),

where  :math:`\iota` is the indicator function

.. math:: \iota_S(x) =
		 \begin{cases}
         1 & \text{if }x \in S, \text{ and } \\
         0 & \text{otherwise}.
		 \end{cases}

For complex systems, these integrals must often be performed using *Monte Carlo* or related methods.  Here, we create a chain of :math:`N` configurations, :math:`X_t`, and calculate the *sample mean*,  [#f1]_

.. math:: \bar{g} = \frac{1}{N} \sum_{t=0}^{N-1}g(X_t).

As :math:`N` approaches infinity, the sample mean converges to :math:`\left \langle g \right \rangle`.

However, for complicated probability densities, this convergence can be particularly slow.  Umbrella sampling overcomes this by instead sampling from multiple biased probability densities,

.. math:: \pi_i(x) = \frac{ \psi_i(x) \pi(x)}{\int  \psi_i(x) \pi(x)dx}.

Here :math`\psi_i` is a collection of :math:`L` nonnegative functions, which we refer to as "bias functions".  In molecular dynamics, gaussian bias functions are commonly used,

.. math:: \psi_i(x) = \exp \left( -\frac{1}{2}k_i \left(q(x)-q^0_i\right)^2/k_B T\right),

where :math:`q` is a collective variable, a function of the coordinates that encodes the large-scale structure of the system and regulates its dynamics.  This choice gives the following biased density.  

.. math:: \pi_i (x) \propto \exp \left[- \left( H_0(x)+\frac{1}{2} k_i \left(q-q^0_i\right)^2\right) / k_B T \right].

This corresponds to adding a harmonic potential centered at :math:`q_i^0` with spring constant :math:`k_i` to the system Hamiltonian.  The simulation corresponding to the :math:`i`'th biasing function is referred to as the simulation "in the :math:`i'th` window."  We then collect a dataseries samplig each biased density, :math:`X_t^i`.  If the bias functions are chosen intelligently, sample means converge much more quickly in the biased densities than in the unbiased probability density.  However, this leaves the task of recombining the data across multiple biased distributions.  **EMUS** is algorithm for performing this task.

EMUS
----
The EMUS algorithm relies on the identity 

.. math:: \left \langle g \right \rangle =   \frac{\sum_{i=1}^L z_i \left \langle g^\ast\right \rangle_i}{\sum_{i=1}^L z_i \left \langle 1^\ast\right \rangle_i}
   :label: eq_core_emus

where we have defined 

.. math:: 
   g^\ast (x) &= \frac{g(x)}{\sum_{k=1}^L \psi_k (x)} \\
   \left \langle g \right \rangle_i &= \int g(x) \pi_i(x) dx .

The weights :math:`z_i` are proportional to the normalization constants of :math:`pi_i`, and solve the equation

.. math::
   z_j = \sum_{i=1}^L z_i F_{ij} \text{, where } F_{ij} = \left \langle \psi_j^\ast \right \rangle_i.
   :label: eq_evec
   
Consequently, the EMUS algorithm works as follows.  For every :math:`i` and each datapoint :math:`X_t^i`, we calculate the estimates

.. math::
   \bar{g}^*_i &= \frac{1}{N_i}\sum_{t=0}^{N_i-1} \frac{g(X_t^i)}{\sum_k \psi_k (X_t^i)}, \\
   \bar{1}^*_i &= \frac{1}{N_i}\sum_{t=0}^{N_i-1} \frac{1}{\sum_k \psi_k (X_t^i)}, \text{ and } \\
   \bar{F}_{ij} &= \frac{1}{N_i}\sum_{t=0}^{N_i-1} \frac{\psi_j(X_t^i)}{\sum_k \psi_k (X_t^i)}.

We then solve our approximation to the eigenvector problem in :eq:`eq_evec` using the matrix of :math:`F_{ij}` values, and compute estimates of :math:`\left \langle g \right \rangle` by using substituting our estimates into :eq:`eq_core_emus`.

.. [#f1] Note that for many common sampling schemes, the variable :math:`t` corresponds to a system "time".
