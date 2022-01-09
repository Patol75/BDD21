#!/usr/bin/env python3
from numpy import array
from scipy.constants import g

from ChemistryData import DM_SS_2004, PM_MO_1995

domainDim = (4e6, 1e6)
stepLoc = (1.25e6, 2.75e6)
stepWidth, conDepth = 2e5, 2e5
oceAge = 7e6 * 365.25 * 8.64e4
surfTemp, mantTemp = 273, 1598.15

k = 3
d = 9e3
rhoH0 = 6e-6
crusHeatProd = 2.6e-13
mantHeatProd = 4e-15

alpha = 3e-5
kappa = 6e-7
rhoMantle = 3.37e3
rhoCraton = 3.3e3
rhoCrust = 2.9e3

c_P = 1187
adGra = alpha * mantTemp * g / c_P

attrib = 'katz_mckenzie_bdd21_'
# Parameters to provide to the melting functions; only parameters described in
# Katz et al. (2003) are valid; parameters names must match these defined in
# __init__ of Katz within Melt.py

# Depleted MORB mantle (eNd = 10)
# melt_inputs = {"alpha_s": alpha, "B1": 1520 + 273.15, "beta2": 1.2,
#                "c_P": c_P, "deltaS": 407, "M_cpx": 0.16783327,
#                "r0": 1.2471514, "r1": -0.17266209, "rho_s": rhoMantle,
#                "X_H2O_bulk": 0.01}

# Equal mix (eNd = 5)
melt_inputs = {"alpha_s": alpha, "B1": 1520 + 273.15, "beta2": 1.2,
               "c_P": c_P, "deltaS": 407, "M_cpx": 0.1702248,
               "r0": 1.09774656, "r1": -0.14365651, "rho_s": rhoMantle,
               "X_H2O_bulk": 0.02}

# Primitive mantle (eNd = 0)
# melt_inputs = {"alpha_s": alpha, "B1": 1520 + 273.15, "beta2": 1.2,
#                "c_P": c_P, "deltaS": 407, "M_cpx": 0.17127832,
#                "r0": 0.99133219, "r1": -0.12357699, "rho_s": rhoMantle,
#                "X_H2O_bulk": 0.028}

# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
# elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
#             'Tm', 'Yb', 'Lu', 'Na', 'Ti', 'Hf', 'Rb', 'Sr', 'Th', 'U', 'Pb',
#             'Nb', 'Zr', 'Y', 'Ta', 'Sc', 'V', 'Cr', 'K', 'P', 'Ba']
elements = ['La', 'Sm', 'Gd', 'Yb', 'Na', 'Ti']
# Value to weigh the contributions of primitive and depleted mantle sources,
# with 0 a pure primitive end-member and 10 its depleted counterpart
eNd = 5
assert 0 <= eNd <= 10
cs_0 = array([(10 - eNd) / 10 * PM_MO_1995[element]
              + eNd / 10 * DM_SS_2004[element] for element in elements])
gnt_out, spl_in = 69e3, 70e3
assert gnt_out <= spl_in

a0 = 1e-13
r0 = 1.94
e0 = 3.5e5
v0 = 6.8e-6

a1 = 1e-13
r1 = 1.94
e1 = 3.5e5
v1 = 6.8e-6

a2 = 1.2e-16
r2 = 1.94
e2 = 3.5e5
v2 = 2.6e-6

lowMtl = 6.6e5

muMax = 1e24
muMin = 1e18
