#!/usr/bin/env python3
from ChemistryData import DM_SS_2004, PM_MS_1995
from numba.typed import Dict
from numba.types import int64, unicode_type
from numpy import asarray
from scipy.constants import g

domain_dim = (1.98e6, 6.6e5)
T_surface, T_mantle = 273.15, 1623.15
vel_x = 2 / 100 / 365.25 / 8.64e4
kappa = 1e-6
rho_mantle = 3300
alpha = 3e-5
c_P = 1187
adiab_grad = alpha * T_mantle * g / c_P

attrib = "katz_mckenzie_bdd21%"
# Value to weigh the contributions of primitive and depleted mantle sources,
# with 0 a pure primitive end-member and 1 its depleted counterpart
src_depletion = 0.9
# Parameters to provide to the melting functions. Most parameters are described in
# Katz et al. (2003); parameters names must match those defined within Melt.py
melt_inputs = {
    "src_depletion": src_depletion,
    "X_H2O_bulk": 0.01,
    "c_P": c_P,
    "α_s": alpha,
    "ρ_s": rho_mantle,
    "ΔS": 407,
}
# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
# fmt: off
elements = [
    "Cs", "Rb", "Ba", "Th", "U", "Nb", "Ta", "La", "Ce", "Pb", "Pr", "Sr", "Nd",
    "Zr", "Hf", "Sm", "Eu", "Gd", "Tb", "Dy", "Y", "Ho", "Er", "Yb", "Lu",
    "Na", "K", "Ti", "P",
    "Li",
    "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn"
]
# Elements for which constant partition coefficients should be used
const_ele = ["Cs", "Rb", "Li"]
# fmt: on
ele_ind_map = Dict.empty(key_type=unicode_type, value_type=int64)
for ind, element in enumerate(elements):
    ele_ind_map[element] = ind
const_ele_ind = asarray([ele_ind_map[element] for element in const_ele])
# Initial concentration of chemical elements in the source
cs_0 = asarray(
    [
        (1 - src_depletion) * PM_MS_1995[element] + src_depletion * DM_SS_2004[element]
        for element in elements
    ]
)
