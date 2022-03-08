#!/usr/bin/env python3
from numpy import asarray
from scipy.constants import g

from ChemistryData import DM_SS_2004, PM_MO_1995

domain_dim = (1.98e6, 6.6e5)
T_surface, T_mantle = 273, 1598
vel_x = 2.1 / 100 / 365.25 / 8.64e4
kappa = 1e-6
rho_mantle = 3300
alpha = 3e-5
c_P = 1187
adiab_grad = alpha * T_mantle * g / c_P

attrib = 'katz_mckenzie_bdd21_'

# Value to weigh the contributions of primitive and depleted mantle sources,
# with 0 a pure primitive end-member and 10 its depleted counterpart
eNd = 10
assert eNd in [0, 5, 10]
if eNd == 10:  # Depleted MORB mantle
    M_cpx, r0, r1, X_H2O_bulk = 0.16783327, 1.2471514, -0.17266209, 0.01
elif eNd == 5:  # Equal mix
    M_cpx, r0, r1, X_H2O_bulk = 0.1702248, 1.09774656, -0.14365651, 0.02
elif eNd == 0:  # Primitive mantle
    M_cpx, r0, r1, X_H2O_bulk = 0.17127832, 0.99133219, -0.12357699, 0.028

# Parameters to provide to the melting functions; only parameters described in
# Katz et al. (2003) are valid; parameters names must match these defined in
# __init__ of Katz within Melt.py
melt_inputs = {"alpha_s": alpha, "B1": 1520 + 273.15, "beta2": 1.2,
               "c_P": c_P, "deltaS": 407, "M_cpx": M_cpx, "r0": r0, "r1": r1,
               "rho_s": rho_mantle, "X_H2O_bulk": X_H2O_bulk}

# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
elements = ['La', 'Sm', 'Gd', 'Yb', 'Na', 'Ti']
# Separate tetravalent ions in two groups
hfse = asarray([elements.index(x)
                for x in [y for y in ['Ti', 'Hf', 'Zr'] if y in elements]])
radio = asarray([elements.index(x)
                for x in [y for y in ['Th', 'U', 'Pb'] if y in elements]])
const = asarray([elements.index(x)
                for x in [y for y in ['Sc', 'Cr', 'Ga'] if y in elements]])

# Initial solid concentration of included elements
cs_0 = asarray([(10 - eNd) / 10 * PM_MO_1995[element]
                + eNd / 10 * DM_SS_2004[element] for element in elements])

# Depths associated with the spinel-garnet transition
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
