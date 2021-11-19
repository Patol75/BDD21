#!/usr/bin/env python3
from numpy import array
from scipy.constants import g

from ChemistryData import DM_SS_2004, PM_MO_1995

domain_dim = (1980e3, 660e3)
T_surface, T_mantle = 273, 1598
vel_x = 2.1 / 100 / 365.25 / 8.64e4
kappa = 1e-6
rho_mantle = 3300
alpha = 3e-5
c_P = 1187
adiab_grad = alpha * T_mantle * g / c_P

attrib = 'katz_mckenzie_bdd21_'
# Parameters to provide to the melting functions; only parameters described in
# Katz et al. (2003) are valid; parameters names must match these defined in
# __init__ of Katz within Melt.py
melt_inputs = {"alpha_s": alpha, "B1": 1520 + 273.15, "beta2": 1.2, "c_P": c_P,
               "deltaS": 407, "M_cpx": 0.18, "r0": 0.94, "r1": -0.1,
               "rho_s": rho_mantle, "X_H2O_bulk": 0.01}
# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Na', 'Ti', 'Hf', 'Rb', 'Sr', 'Th', 'U', 'Pb',
            'Nb', 'Zr', 'Y', 'Ta', 'Sc', 'V', 'Cr', 'K', 'P', 'Ba']
eNd = 10
assert 0 <= eNd <= 10
cs_0 = array([(10 - eNd) / 10 * PM_MO_1995[element]
              + eNd / 10 * DM_SS_2004[element] for element in elements])
gnt_out, spl_in = 69e3, 70e3
assert gnt_out <= spl_in

# Rheology:
mu_max, mu_min = 1e25, 1e18

# Diffusion creep
Ediff_UM = 3e5
Vdiff_UM = 4e-6
Adiff_UM = 3e-11

# Dislocationn creep
n = 3.5
Adisl_UM = 5e-16
Edisl_UM = 540e3
Vdisl_UM = 16e-6
