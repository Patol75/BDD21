#!/usr/bin/env python3
from numpy import asarray
from scipy.constants import g

from ChemistryData import DM_SS_2004, PM_MS_1995

domain_dim = (1.98e6, 6.6e5)
T_mantle = 1598
rho_mantle = 3300
alpha = 3e-5
c_P = 1187
adiab_grad = alpha * T_mantle * g / c_P
depth_lab = 9e4

attrib = "katz_mckenzie_bdd21_"
# Value to weigh the contributions of primitive and depleted mantle sources,
# with 0 a pure primitive end-member and 10 its depleted counterpart
ɛNd = 0
# Parameters to provide to the melting functions. Most parameters are described in
# Katz et al. (2003); parameters names must match those defined within Melt.py
melt_inputs = {
    "ɛNd": ɛNd,
    "X_H2O_bulk": 0.028,
    "c_P": c_P,
    "α_s": alpha,
    "ρ_s": rho_mantle,
    "ΔS": 407,
}
# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
elements = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Na", "Ti", "Hf", "Rb", "Sr", "Th", "U", "Pb", "Nb", "Zr", "Y",
    "Ta", "Sc", "V", "Cr", "K", "P", "Ba",
]
# Separate tetravalent ions in two groups
hfse = asarray(
    [elements.index(x) for x in [y for y in ["Ti", "Hf", "Zr"] if y in elements]]
)
radio = asarray(
    [elements.index(x) for x in [y for y in ["Th", "U", "Pb"] if y in elements]]
)
const = asarray(
    [elements.index(x) for x in [y for y in ["Sc", "Cr", "Ga"] if y in elements]]
)
# Initial solid concentration of included elements
cs_0 = asarray(
    [
        (10 - ɛNd) / 10 * PM_MS_1995[element] + ɛNd / 10 * DM_SS_2004[element]
        for element in elements
    ]
)
