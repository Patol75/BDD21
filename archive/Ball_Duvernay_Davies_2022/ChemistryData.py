#!/usr/bin/env python3
from numba import float64
from numba.typed import Dict
from numba.types import unicode_type
from numpy import array, nan

valence = {}
for element in ["Cs", "K", "Na", "Rb"]:
    valence[element] = 1
for element in ["Ba", "Ca", "Co", "Fe", "Mg", "Mn", "Ni", "Ra", "Sr"]:
    valence[element] = 2
for element in [
    "Ce",
    "Cr",
    "Dy",
    "Er",
    "Eu",
    "Ga",
    "Gd",
    "Ho",
    "La",
    "Lu",
    "Nd",
    "Pr",
    "Sc",
    "Sm",
    "Tb",
    "Tm",
    "Y",
    "Yb",
]:
    valence[element] = 3
for element in ["Hf", "Pb", "Th", "Ti", "U", "Zr"]:
    valence[element] = 4
for element in ["Nb", "P", "Ta", "V"]:
    valence[element] = 5

radii = {}
# Shannon - Acta Crystallographica (1976)
# Updated values from Wood and Blundy - Treatise on Geochemistry (2014)
radii["Ba"], radii["Ce"] = array([1.35, 1.42]), array([1.01, 1.143])
radii["Co"], radii["Cr"] = array([0.745, 0.9]), array([0.615, nan])
radii["Cs"], radii["Dy"] = array([1.67, 1.74]), array([0.912, 1.027])
radii["Er"], radii["Eu"] = array([0.89, 1.004]), array([0.947, 1.066])
radii["Ga"], radii["Gd"] = array([0.62, nan]), array([0.938, 1.053])
radii["Hf"], radii["Ho"] = array([0.71, 0.83]), array([0.901, 1.015])
radii["K"], radii["La"] = array([1.38, 1.51]), array([1.032, 1.16])
radii["Lu"], radii["Mn"] = array([0.861, 0.977]), array([0.83, 0.96])
radii["Na"], radii["Nb"] = array([1.02, 1.18]), array([0.64, 0.74])
radii["Nd"], radii["Ni"] = array([0.983, 1.109]), array([0.69, nan])
radii["P"], radii["Pb"] = array([0.38, nan]), array([0.775, 0.94])
radii["Pr"], radii["Ra"] = array([0.99, 1.126]), array([nan, 1.48])
radii["Rb"], radii["Sc"] = array([1.52, 1.61]), array([0.745, 0.87])
radii["Sm"], radii["Sr"] = array([0.958, 1.079]), array([1.18, 1.26])
radii["Ta"], radii["Tb"] = array([0.64, 0.74]), array([0.923, 1.04])
radii["Th"], radii["Ti"] = array([0.94, 1.041]), array([0.605, 0.74])
radii["Tm"], radii["U"] = array([0.88, 0.994]), array([0.89, 0.983])
radii["V"], radii["Y"] = array([0.54, nan]), array([0.9, 1.019])
radii["Yb"], radii["Zr"] = array([0.868, 0.985]), array([0.72, 0.84])

part_coeff = {}
# McKenzie and O'Nions - Journal of Petrology (1995)
part_coeff["Na"] = array([1e-5, 0.05, 0.2, 0.39, 0.0, 0.04])
part_coeff["P"] = array([0.0, 0.03, 0.03, 0.0, 0.0, 0.1])
part_coeff["K"] = array([1.8e-4, 1e-3, 2e-3, 0.18, 1e-4, 1e-3])
part_coeff["Sc"] = array([0.16, 0.33, 0.51, 0.02, 0.0, 2.27])
part_coeff["Ti"] = array([0.02, 0.1, 0.18, 0.04, 0.15, 0.28])
part_coeff["V"] = array([0.06, 0.9, 1.31, 0.0, 0.0, 1.57])
part_coeff["Cr"] = array([0.3, 1.5, 3.0, 0.05, 300.0, 5.5])
part_coeff["Mn"] = array([0.5, 0.7, 0.44, 0.05, 0.25, 2.05])
part_coeff["Co"] = array([1.0, 2.0, 2.0, 0.05, 2.0, 2.0])
part_coeff["Ni"] = array([9.4, 9.4, 9.4, 0.0, 0.0, 0.0])
part_coeff["Ga"] = array([0.04, 0.2, 0.74, 0.0, 5.0, 5.0])
part_coeff["Rb"] = array([1.8e-4, 6e-4, 1e-3, 0.03, 1e-4, 7e-4])
part_coeff["Sr"] = array([1.9e-4, 7e-3, 0.13, 2.0, 0.0, 1.1e-3])
part_coeff["Y"] = array([5e-3, 5e-3, 0.2, 0.03, 0.0, 2.11])
part_coeff["Zr"] = array([0.01, 0.03, 0.1, 0.01, 0.0, 0.32])
part_coeff["Nb"] = array([5e-3, 5e-3, 0.02, 0.01, 0.0, 0.07])
part_coeff["Cs"] = array([5e-5, 1e-4, 2e-4, 0.025, 1e-4, 2e-4])
part_coeff["Ba"] = array([3e-4, 1e-4, 5e-4, 0.33, 1e-4, 5e-4])
part_coeff["Hf"] = array([0.01, 0.01, 0.22, 0.01, 0.0, 0.44])
part_coeff["Ta"] = array([5e-3, 5e-3, 0.02, 0.0, 0.0, 0.04])
part_coeff["Pb"] = array([1e-4, 1.3e-3, 0.01, 0.36, 0.0, 5e-4])
part_coeff["Th"] = array([1e-4, 1e-4, 2.6e-4, 0.05, 0.0, 1e-4])
# McKenzie and O'Nions - Journal of Petrology (1991)
part_coeff["La"] = array([4e-4, 2e-3, 0.054, 0.27, 0.01, 0.01])
part_coeff["Ce"] = array([5e-4, 3e-3, 0.098, 0.2, 0.01, 0.021])
part_coeff["Pr"] = array([8e-4, 4.8e-3, 0.15, 0.17, 0.01, 0.054])
part_coeff["Nd"] = array([1e-3, 6.8e-3, 0.21, 0.14, 0.01, 0.087])
part_coeff["Sm"] = array([1.3e-3, 0.01, 0.26, 0.11, 0.01, 0.217])
part_coeff["Eu"] = array([1.6e-3, 0.013, 0.31, 0.73, 0.01, 0.32])
part_coeff["Gd"] = array([1.5e-3, 0.016, 0.3, 0.066, 0.01, 0.498])
part_coeff["Tb"] = array([1.5e-3, 0.019, 0.31, 0.06, 0.01, 0.75])
part_coeff["Dy"] = array([1.7e-3, 0.022, 0.33, 0.055, 0.01, 1.06])
part_coeff["Ho"] = array([1.6e-3, 0.026, 0.31, 0.048, 0.01, 1.53])
part_coeff["Er"] = array([1.5e-3, 0.03, 0.3, 0.041, 0.01, 2.0])
part_coeff["Tm"] = array([1.5e-3, 0.04, 0.29, 0.036, 0.01, 3.0])
part_coeff["Yb"] = array([1.5e-3, 0.049, 0.28, 0.031, 0.01, 4.03])
part_coeff["Lu"] = array([1.5e-3, 0.06, 0.28, 0.025, 0.01, 5.5])
part_coeff["U"] = array([1e-4, 1e-4, 3.6e-4, 0.11, 0.0, 1e-4])
# Code from McKenzie and O'Nions - Journal of Petrology (1995)
part_coeff["Li"] = array([0.0, 0.0, 0.59, 0.0, 0.0, 0.0])
part_coeff["Cu"] = array([0.0, 0.0, 0.36, 0.0, 0.0, 0.0])
part_coeff["Zn"] = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Salters and Stracke - Geochemistry, Geophysics, Geosystems (2004)
DM_SS_2004 = {
    "La": 0.234,
    "Ce": 0.772,
    "Pr": 0.131,
    "Nd": 0.713,
    "Sm": 0.27,
    "Eu": 0.107,
    "Gd": 0.395,
    "Tb": 0.075,
    "Dy": 0.531,
    "Ho": 0.122,
    "Er": 0.371,
    "Tm": 0.06,
    "Yb": 0.401,
    "Lu": 0.063,
    "Hf": 0.199,
    "Rb": 0.088,
    "Sr": 9.8,
    "Th": 0.0137,
    "U": 4.7e-3,
    "Pb": 0.0232,
    "Nb": 0.21,
    "Ti": 798,
    "Zr": 7.94,
    "Y": 4.07,
    "Ta": 0.0138,
    "Li": 0.7,
    "Sc": 16.3,
    "V": 79,
    "Cr": 2500,
    "Ni": 1960,
    "Na": 2151.4,
    "K": 60,
    "Mn": 1045,
    "P": 40.7,
    "Co": 106,
    "Ba": 1.2,
    "Ga": 3.2,
    "Cu": 30,
    "Zn": 56,
    "Cs": 1.32e-3,
}

# McDonough and Sun - Chemical Geology (1995)
PM_MS_1995 = {
    "La": 0.648,
    "Ce": 1.675,
    "Pr": 0.254,
    "Nd": 1.250,
    "Sm": 0.406,
    "Eu": 0.154,
    "Gd": 0.544,
    "Tb": 0.099,
    "Dy": 0.674,
    "Ho": 0.149,
    "Er": 0.438,
    "Tm": 0.068,
    "Yb": 0.441,
    "Lu": 0.0675,
    "Hf": 0.283,
    "Rb": 0.600,
    "Sr": 19.9,
    "Th": 0.0795,
    "U": 0.0203,
    "Pb": 0.150,
    "Nb": 0.658,
    "Ti": 1205,
    "Zr": 10.5,
    "Y": 4.30,
    "Ta": 0.037,
    "Li": 1.6,
    "Sc": 16.2,
    "V": 82,
    "Cr": 2625,
    "Ni": 1960,
    "Na": 2670,
    "K": 240,
    "Mn": 1045,
    "P": 90,
    "Co": 105,
    "Ba": 6.6,
    "Ga": 4.0,
    "Cu": 30,
    "Zn": 55,
    "Cs": 0.021,
}

# Na to U: McKenzie and O'Nions - Journal of Petrology (1995)
# La to Lu: McKenzie and O'Nions - Journal of Petrology (1991)
# Li to Ra: Code from McKenzie and O'Nions - Journal of Petrology (1995)
PM_MO_1995 = {
    "Na": 1800,
    "P": 61,
    "K": 200,
    "Sc": 12,
    "Ti": 1020,
    "V": 103,
    "Cr": 3000,
    "Co": 105,
    "Ni": 2000,
    "Ga": 3.7,
    "Rb": 0.62,
    "Sr": 20,
    "Y": 3.45,
    "Zr": 8.51,
    "Nb": 0.54,
    "Cs": 0.01,
    "Ba": 6.5,
    "Hf": 0.25,
    "Ta": 0.031,
    "Pb": 0.155,
    "Th": 0.07,
    "U": 0.018,
    "La": 0.55,
    "Ce": 1.4,
    "Pr": 0.22,
    "Nd": 1.08,
    "Sm": 0.35,
    "Eu": 0.13,
    "Gd": 0.457,
    "Tb": 0.084,
    "Dy": 0.57,
    "Ho": 0.13,
    "Er": 0.372,
    "Tm": 0.058,
    "Yb": 0.372,
    "Lu": 0.057,
    "Li": 2.7,
    "Mn": 1000,
    "Cu": 40,
    "Zn": 68,
    "Ra": 6.38e-9,
}

# Ball, Duvernay and Davies - Geochemistry, Geophysics, Geosystems (2022)
mnrl_mode_coeff = Dict.empty(key_type=unicode_type, value_type=float64[:, :])
mnrl_mode_coeff["ol_spl"] = array([[-0.115, 0.031, 0.318], [-0.039, 0.126, 0.419]])
mnrl_mode_coeff["ol_gnt"] = array([[0.048, -0.558, 1.298], [-0.003, 0.035, 0.445]])
mnrl_mode_coeff["cpx"] = array([[0.037, -0.229, -0.606], [-0.011, 0.112, 0.058]])
mnrl_mode_coeff["spl"] = array([[0.026, -0.013, -0.087], [-0.004, 0.004, 0.02]])
mnrl_mode_coeff["gnt"] = array([[-0.005, 0.078, -0.557], [-0.001, 0.033, 0.008]])
