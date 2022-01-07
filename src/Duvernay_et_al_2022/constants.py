#!/usr/bin/env python3
from numpy import array
from ChemistryData import DM_SS_2004, PM_MO_1995

domain_dim = (1980e3, 660e3)
T_mantle = 1598
rho_mantle = 3300
alpha = 3e-5
c_P = 1187

attrib = 'katz_mckenzie_bdd21_'
# Parameters to provide to the melting functions; only parameters described in
# Katz et al. (2003) are valid; parameters names must match these defined in
# __init__ of Katz within Melt.py
melt_inputs = {"alpha_s": alpha, "c_P": c_P, "deltaS": 407,
               "rho_s": rho_mantle, "X_H2O_bulk": 0.028}
# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Na', 'Ti', 'Hf', 'Rb', 'Sr', 'Th', 'U', 'Pb',
            'Nb', 'Zr', 'Y', 'Ta', 'Sc', 'V', 'Cr', 'K', 'P', 'Ba']
# Value to weigh the contributions of primitive and depleted mantle sources,
# with 0 a pure primitive end-member and 10 its depleted counterpart
eNd = 0
assert 0 <= eNd <= 10
# Initial solid concentration of included elements
cs_0 = array([(10 - eNd) / 10 * PM_MO_1995[element]
              + eNd / 10 * DM_SS_2004[element] for element in elements])
