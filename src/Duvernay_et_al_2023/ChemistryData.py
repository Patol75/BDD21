#!/usr/bin/env python3
from numba.typed import Dict
from numba.types import float64, unicode_type
from numpy import array, isfinite, nan

valence = {}
for element in ["Cs", "K", "Li", "Na", "Rb"]:
    valence[element] = 1
for element in ["Ba", "Co", "Cu", "Ni", "Pb", "Sr", "Zn"]:
    valence[element] = 2
# fmt: off
for element in [
    "Ce", "Cr", "Dy", "Er", "Eu", "Gd", "Ho", "La",
    "Lu", "Nd", "Pr", "Sc", "Sm", "Tb", "Y", "Yb"
]:
    valence[element] = 3
# fmt: on
for element in ["Hf", "Th", "Ti", "U", "Zr"]:
    valence[element] = 4
for element in ["Nb", "P", "Ta", "V"]:
    valence[element] = 5

radii = {}
# Shannon - Acta Crystallographica (1976)
# Updated values from
# Blundy and Wood - Reviews in Mineralogy and Geochemistry (2003)
# Wood and Blundy - Treatise on Geochemistry (2014)
radii["Ba"], radii["Ce"] = array([1.35, 1.42]), array([1.01, 1.143])
radii["Co"], radii["Cr"] = array([0.745, 0.9]), array([0.615, nan])
radii["Cs"], radii["Cu"] = array([1.67, 1.74]), array([0.73, nan])
radii["Dy"] = array([0.912, 1.027])
radii["Er"], radii["Eu"] = array([0.89, 1.004]), array([0.947, 1.066])
radii["Ga"], radii["Gd"] = array([0.62, nan]), array([0.938, 1.053])
radii["Hf"], radii["Ho"] = array([0.71, 0.83]), array([0.901, 1.015])
radii["K"], radii["La"] = array([1.38, 1.51]), array([1.032, 1.16])
radii["Li"] = array([0.76, 0.92])
radii["Lu"], radii["Mn"] = array([0.861, 0.977]), array([0.83, 0.96])
radii["Na"], radii["Nb"] = array([1.02, 1.18]), array([0.64, 0.74])
radii["Nd"], radii["Ni"] = array([0.983, 1.109]), array([0.69, nan])
radii["P"], radii["Pb"] = array([0.38, nan]), array([1.19, 1.29])
radii["Pr"], radii["Ra"] = array([0.99, 1.126]), array([nan, 1.48])
radii["Rb"], radii["Sc"] = array([1.52, 1.61]), array([0.745, 0.87])
radii["Sm"], radii["Sr"] = array([0.958, 1.079]), array([1.18, 1.26])
radii["Ta"], radii["Tb"] = array([0.64, 0.74]), array([0.923, 1.04])
radii["Th"], radii["Ti"] = array([0.919, 1.041]), array([0.605, 0.74])
radii["Tm"], radii["U"] = array([0.88, 0.994]), array([0.875, 0.983])
radii["V"], radii["Y"] = array([0.54, nan]), array([0.9, 1.019])
radii["Yb"], radii["Zn"] = array([0.868, 0.985]), array([0.74, 0.90])
radii["Zr"] = array([0.72, 0.84])

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
part_coeff["Hf"] = array([0.0, 0.01, 0.22, 0.01, 0.0, 0.44])
part_coeff["Ta"] = array([5e-3, 5e-3, 0.02, 0.0, 0.0, 0.04])
part_coeff["Pb"] = array([1e-4, 1.3e-3, 0.01, 0.36, 0.0, 5e-4])
part_coeff["Th"] = array([1e-4, 1e-4, 2.6e-4, 0.05, 0.0, 1e-4])
part_coeff["U"] = array([1e-4, 1e-4, 3.6e-4, 0.11, 0.0, 1e-4])
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

# Warren - Lithos (2016)
part_coeff_warren = {}  # Table S3
part_coeff_warren["La"] = array([2.3e-6, 7e-4, 0.055, nan, nan, nan])
part_coeff_warren["Ce"] = array([7.3e-6, 1.6e-3, 0.0876, nan, nan, nan])
part_coeff_warren["Pr"] = array([2.1e-5, 3.2e-3, 0.1318, nan, nan, nan])
part_coeff_warren["Nd"] = array([5.8e-5, 6.0e-3, 0.1878, nan, nan, nan])
part_coeff_warren["Sm"] = array([2.9e-4, 0.0158, 0.3083, nan, nan, nan])
part_coeff_warren["Eu"] = array([5.5e-4, 0.0227, 0.3638, nan, nan, nan])
part_coeff_warren["Gd"] = array([1.0e-3, 0.0315, 0.4169, nan, nan, nan])
part_coeff_warren["Tb"] = array([1.7e-3, 0.0422, 0.4645, nan, nan, nan])
part_coeff_warren["Dy"] = array([2.9e-3, 0.0549, 0.5034, nan, nan, nan])
part_coeff_warren["Y"] = array([3.9e-3, 0.0635, 0.5219, nan, nan, nan])
part_coeff_warren["Ho"] = array([4.5e-3, 0.0680, 0.5294, nan, nan, nan])
part_coeff_warren["Er"] = array([6.6e-3, 0.0808, 0.5437, nan, nan, nan])
part_coeff_warren["Tm"] = array([9.2e-3, 0.0928, 0.5482, nan, nan, nan])
part_coeff_warren["Yb"] = array([0.0121, 0.1036, 0.5453, nan, nan, nan])
part_coeff_warren["Lu"] = array([0.0153, 0.1128, 0.5373, nan, nan, nan])
# Le Roux et al. - American Mineralogist (2015)
part_coeff_leroux = {}
part_coeff_leroux["Cu"] = array([0.13, 0.12, 0.09, nan, 0.25, 0.042])
part_coeff_leroux["Ga"] = array([0.026, 0.38, 0.37, nan, 6.5, 0.39])
part_coeff_leroux["Ge"] = array([0.43, 0.87, 0.87, nan, 0.4, 1.51])
part_coeff_leroux["Ti"] = array([8e-3, 0.0656, 0.124, nan, 0.084, 0.262])
part_coeff_leroux["Sc"] = array([0.15, 0.495, 0.84, nan, 0.058, 5.98])
part_coeff_leroux["V"] = array([0.14, 1.06, 1.48, nan, 2.75, 1.84])
part_coeff_leroux["Cr"] = array([0.79, 8.8, 7.5, nan, 54.0, 10.2])
part_coeff_leroux["Mn"] = array([0.781, 0.640, 0.768, nan, 0.46, 1.241])
part_coeff_leroux["Zn"] = array([0.96, 0.451, 0.333, nan, 5.2, 0.213])
part_coeff_leroux["Fe"] = array([1.034, 0.55, 0.49, nan, 0.95, 0.654])
part_coeff_leroux["Co"] = array([2.37, 1.29, 0.86, nan, 3.0, 0.83])
part_coeff_leroux["Ni"] = array([6.2, 3.7, 22.0, nan, 10.0, 8.0])
# Wijbrans et al. - Contributions to Mineralogy and Petrology (2015)
part_coeff_wijbrans = {}  # Sample sp5_02
part_coeff_wijbrans["Y"] = array([nan, nan, nan, nan, 2e-4, nan])
part_coeff_wijbrans["Zr"] = array([nan, nan, nan, nan, 4e-4, nan])
part_coeff_wijbrans["Nb"] = array([nan, nan, nan, nan, 2e-4, nan])
part_coeff_wijbrans["Mo"] = array([nan, nan, nan, nan, 1.1e-3, nan])
part_coeff_wijbrans["Rh"] = array([nan, nan, nan, nan, 0.051, nan])
part_coeff_wijbrans["Lu"] = array([nan, nan, nan, nan, 2.8e-4, nan])
part_coeff_wijbrans["Hf"] = array([nan, nan, nan, nan, 1e-3, nan])
part_coeff_wijbrans["Ta"] = array([nan, nan, nan, nan, 2.7e-4, nan])
part_coeff_wijbrans["W"] = array([nan, nan, nan, nan, 1.8e-4, nan])
part_coeff_wijbrans["Pt"] = array([nan, nan, nan, nan, 0.014, nan])
part_coeff_wijbrans["Th"] = array([nan, nan, nan, nan, 1e-5, nan])
part_coeff_wijbrans["U"] = array([nan, nan, nan, nan, 3e-5, nan])
# Fulmer et al. - Geochimica et Cosmochimica Acta (2010)
part_coeff_fulmer = {}
part_coeff_fulmer["Li"] = array([nan, nan, nan, nan, nan, 8e-3])
part_coeff_fulmer["Be"] = array([nan, nan, nan, nan, nan, 7e-3])
# part_coeff_fulmer["Rb"] = array([nan, nan, nan, nan, nan, 0.11])
# part_coeff_fulmer["Sr"] = array([nan, nan, nan, nan, nan, 0.025])
part_coeff_fulmer["Y"] = array([nan, nan, nan, nan, nan, 3.8])
part_coeff_fulmer["Zr"] = array([nan, nan, nan, nan, nan, 0.22])
part_coeff_fulmer["Nb"] = array([nan, nan, nan, nan, nan, 7e-4])
part_coeff_fulmer["Mo"] = array([nan, nan, nan, nan, nan, 9e-3])
part_coeff_fulmer["Sn"] = array([nan, nan, nan, nan, nan, 0.03])
# part_coeff_fulmer["Cs"] = array([nan, nan, nan, nan, nan, 0.16])
# part_coeff_fulmer["Ba"] = array([nan, nan, nan, nan, nan, 0.022])
# part_coeff_fulmer["La"] = array([nan, nan, nan, nan, nan, 0.011])
# part_coeff_fulmer["Ce"] = array([nan, nan, nan, nan, nan, 9e-3])
part_coeff_fulmer["Pr"] = array([nan, nan, nan, nan, nan, 0.018])
part_coeff_fulmer["Nd"] = array([nan, nan, nan, nan, nan, 0.041])
part_coeff_fulmer["Sm"] = array([nan, nan, nan, nan, nan, 0.221])
part_coeff_fulmer["Eu"] = array([nan, nan, nan, nan, nan, 0.40])
part_coeff_fulmer["Gd"] = array([nan, nan, nan, nan, nan, 0.76])
part_coeff_fulmer["Tb"] = array([nan, nan, nan, nan, nan, 1.38])
part_coeff_fulmer["Dy"] = array([nan, nan, nan, nan, nan, 2.32])
part_coeff_fulmer["Ho"] = array([nan, nan, nan, nan, nan, 3.51])
part_coeff_fulmer["Er"] = array([nan, nan, nan, nan, nan, 4.9])
part_coeff_fulmer["Tm"] = array([nan, nan, nan, nan, nan, 6.1])
part_coeff_fulmer["Yb"] = array([nan, nan, nan, nan, nan, 6.9])
part_coeff_fulmer["Lu"] = array([nan, nan, nan, nan, nan, 7.7])
part_coeff_fulmer["Hf"] = array([nan, nan, nan, nan, nan, 0.216])
part_coeff_fulmer["Ta"] = array([nan, nan, nan, nan, nan, 1.1e-3])
part_coeff_fulmer["W"] = array([nan, nan, nan, nan, nan, 0.0125])
part_coeff_fulmer["Tl"] = array([nan, nan, nan, nan, nan, 0.3])
part_coeff_fulmer["Pb"] = array([nan, nan, nan, nan, nan, 0.013])
# part_coeff_fulmer["Th"] = array([nan, nan, nan, nan, nan, 0.018])
# part_coeff_fulmer["U"] = array([nan, nan, nan, nan, nan, 0.010])
# Frei et al. - Contributions to Mineralogy and Petrology (2009)
part_coeff_frei = {}  # Run # 6617-05-02
part_coeff_frei["Li"] = array([nan, 0.22, nan, nan, nan, nan])
part_coeff_frei["Be"] = array([nan, 0.0180, nan, nan, nan, nan])
part_coeff_frei["B"] = array([nan, 0.021, nan, nan, nan, nan])
part_coeff_frei["K"] = array([nan, 2.9e-3, nan, nan, nan, nan])
part_coeff_frei["Rb"] = array([nan, 1.4e-4, nan, nan, nan, nan])
part_coeff_frei["Sr"] = array([nan, 3.6e-3, nan, nan, nan, nan])
part_coeff_frei["Zr"] = array([nan, 0.028, nan, nan, nan, nan])
part_coeff_frei["Nb"] = array([nan, 1.4e-3, nan, nan, nan, nan])
part_coeff_frei["Cs"] = array([nan, 4e-5, nan, nan, nan, nan])
part_coeff_frei["Ba"] = array([nan, 2.4e-5, nan, nan, nan, nan])
part_coeff_frei["Hf"] = array([nan, 0.043, nan, nan, nan, nan])
part_coeff_frei["Ta"] = array([nan, 4.9e-3, nan, nan, nan, nan])
part_coeff_frei["Pb"] = array([nan, 4.9e-3, nan, nan, nan, nan])
part_coeff_frei["Th"] = array([nan, 1e-3, nan, nan, nan, nan])
part_coeff_frei["U"] = array([nan, 2e-3, nan, nan, nan, nan])
# Adam and Green - Contributions to Mineralogy and Petrology (2006)
part_coeff_adam = {}  # Run R77 (ol), 1948 (cpx), and 1955 (grt)
part_coeff_adam["Li"] = array([0.29, nan, 0.23, nan, nan, nan])
part_coeff_adam["Be"] = array([1e-3, nan, 0.045, nan, nan, nan])
part_coeff_adam["B"] = array([0.02, nan, 5e-3, nan, nan, 6e-3])
part_coeff_adam["Rb"] = array([4e-4, nan, 4e-4, nan, nan, nan])
part_coeff_adam["Sr"] = array([nan, nan, 0.098, nan, nan, 5e-4])
part_coeff_adam["Zr"] = array([1e-3, nan, 0.071, nan, nan, nan])
part_coeff_adam["Nb"] = array([7e-5, nan, 2e-3, nan, nan, nan])
part_coeff_adam["Cs"] = array([2e-4, nan, 2e-4, nan, nan, nan])
part_coeff_adam["Ba"] = array([1e-4, nan, 6e-4, nan, nan, 2e-4])
part_coeff_adam["La"] = array([nan, nan, nan, nan, nan, 5e-4])
part_coeff_adam["Ce"] = array([nan, nan, nan, nan, nan, 2.7e-3])
part_coeff_adam["Hf"] = array([8e-4, nan, 0.15, nan, nan, nan])
part_coeff_adam["Ta"] = array([2e-4, nan, 5.2e-3, nan, nan, nan])
part_coeff_adam["Pb"] = array([1e-3, nan, 0.014, nan, nan, 0.02])
part_coeff_adam["Th"] = array([1.7e-4, nan, 3.5e-3, nan, nan, 8e-4])
part_coeff_adam["U"] = array([2.6e-4, nan, 5.5e-3, nan, nan, 3.6e-3])
# McDade et al. - Physics of the Earth and Planetary Interiors (2003)
part_coeff_mcdade = {}  # Tinaquillo Lherzolite
part_coeff_mcdade["Na"] = array([nan, 0.056, 0.223, nan, nan, nan])
part_coeff_mcdade["Sr"] = array([1.8e-4, nan, nan, nan, nan, nan])
# Zanetti et al. - Lithos (2004)
part_coeff_zanetti = {}  # T3-1055
part_coeff_zanetti["Na"] = array([6.5e-3, nan, nan, nan, nan, nan])
part_coeff_zanetti["K"] = array([0.013, nan, nan, nan, nan, nan])
# Pertermann et al. - Geochemistry, Geophysics, Geosystems (2004)
part_coeff_pertermann = {}  # A343
part_coeff_pertermann["Na"] = array([nan, nan, nan, nan, nan, 0.0435])
# Klemme et al. - Geochemistry, Geophysics, Geosystems (2002)
part_coeff_klemme = {}
part_coeff_klemme["K"] = array([nan, nan, 7e-3, nan, nan, 3e-4])
part_coeff_klemme["Rb"] = array([nan, nan, nan, nan, nan, 7e-4])
part_coeff_klemme["Cs"] = array([nan, nan, nan, nan, nan, 1e-4])
# Brunet and Chazot - Chemical Geology (2001)
part_coeff_brunet = {}
part_coeff_brunet["P"] = array([0.1, 0.03, 0.05, nan, 0.0, 0.15])

for part_coeff_dict in [
    part_coeff_warren,
    part_coeff_leroux,
    part_coeff_wijbrans,
    part_coeff_fulmer,
    part_coeff_frei,
    part_coeff_adam,
    part_coeff_mcdade,
    part_coeff_zanetti,
    part_coeff_pertermann,
    part_coeff_klemme,
    part_coeff_brunet,
]:
    for key, value in part_coeff_dict.items():
        ind = isfinite(value)
        try:
            part_coeff[key][ind] = value[ind]
        except KeyError:
            part_coeff[key] = value
for key, value in part_coeff.items():
    ind = isfinite(value)
    part_coeff[key][~ind] = 0.0


# Salters and Stracke - Geochemistry, Geophysics, Geosystems (2004)
# fmt: off
DM_SS_2004 = {
    "La": 0.234, "Ce": 0.772, "Pr": 0.131, "Nd": 0.713, "Sm": 0.27,
    "Eu": 0.107, "Gd": 0.395, "Tb": 0.075, "Dy": 0.531, "Ho": 0.122,
    "Er": 0.371, "Tm": 0.06, "Yb": 0.401, "Lu": 0.063, "Hf": 0.199,
    "Rb": 0.088, "Sr": 9.8, "Th": 0.0137, "U": 4.7e-3, "Pb": 0.0232,
    "Nb": 0.21, "Ti": 798, "Zr": 7.94, "Y": 4.07, "Ta": 0.0138,
    "Li": 0.7, "Sc": 16.3, "V": 79, "Cr": 2500, "Ni": 1960,
    "Na": 2151.4, "K": 60, "Mn": 1045, "P": 40.7, "Co": 106,
    "Ba": 1.2, "Ga": 3.2, "Cu": 30, "Zn": 56, "Cs": 1.32e-3
}
# fmt: on

# Workman and Hart - Earth and Planetary Science Letters (2005)
# fmt: off
DM_WH_2005 = {
    "La": 0.192, "Ce": 0.550, "Pr": 0.107, "Nd": 0.581, "Sm": 0.239,
    "Eu": 0.096, "Gd": 0.358, "Tb": 0.070, "Dy": 0.505, "Ho": 0.115,
    "Er": 0.348, "Tm": nan, "Yb": 0.365, "Lu": 0.058, "Hf": 0.157,
    "Rb": 0.050, "Sr": 7.664, "Th": 0.0079, "U": 3.2e-3, "Pb": 0.018,
    "Nb": 0.1485, "Ti": 716.3, "Zr": 5.082, "Y": 3.328, "Ta": 9.6e-3,
    "Li": nan, "Sc": nan, "V": nan, "Cr": nan, "Ni": nan,
    "Na": nan, "K": nan, "Mn": nan, "P": nan, "Co": nan,
    "Ba": 0.563, "Ga": nan, "Cu": nan, "Zn": nan, "Cs": nan
}
# fmt: on

# Boyet and Carlson - Earth and Planetary Science Letters (2006)
# fmt: off
DM_BC_2006 = {
    "La": 0.26, "Ce": 0.87, "Pr": nan, "Nd": 0.81, "Sm": 0.29,
    "Eu": 0.12, "Gd": 0.42, "Tb": nan, "Dy": 0.58, "Ho": nan,
    "Er": 0.39, "Tm": nan, "Yb": 0.40, "Lu": 0.063, "Hf": 0.22,
    "Rb": 0.1, "Sr": 11.1, "Th": 0.016, "U": 5.4e-3, "Pb": 0.035,
    "Nb": 0.24, "Ti": nan, "Zr": nan, "Y": nan, "Ta": nan,
    "Li": nan, "Sc": nan, "V": nan, "Cr": nan, "Ni": nan,
    "Na": nan, "K": 68.4, "Mn": nan, "P": nan, "Co": nan,
    "Ba": 1.37, "Ga": nan, "Cu": nan, "Zn": nan, "Cs": nan
}
# fmt: on

# McDonough and Sun - Chemical Geology (1995)
# fmt: off
PM_MS_1995 = {
    "La": 0.648, "Ce": 1.675, "Pr": 0.254, "Nd": 1.250, "Sm": 0.406,
    "Eu": 0.154, "Gd": 0.544, "Tb": 0.099, "Dy": 0.674, "Ho": 0.149,
    "Er": 0.438, "Tm": 0.068, "Yb": 0.441, "Lu": 0.0675, "Hf": 0.283,
    "Rb": 0.600, "Sr": 19.9, "Th": 0.0795, "U": 0.0203, "Pb": 0.150,
    "Nb": 0.658, "Ti": 1205, "Zr": 10.5, "Y": 4.30, "Ta": 0.037,
    "Li": 1.6, "Sc": 16.2, "V": 82, "Cr": 2625, "Ni": 1960,
    "Na": 2670, "K": 240, "Mn": 1045, "P": 90, "Co": 105,
    "Ba": 6.6, "Ga": 4.0, "Cu": 30, "Zn": 55, "Cs": 0.021
}
# fmt: on

# Yaxley and Sobolev - Contributions to Mineralogy and Petrology (2007)
# fmt: off
GB_YS_2007 = {
    "La": 0.56, "Ce": 1.90, "Pr": 0.37, "Nd": 2.73, "Sm": 1.02,
    "Eu": 0.58, "Gd": 1.52, "Tb": 0.28, "Dy": 1.94, "Ho": 0.40,
    "Er": 1.14, "Tm": 0.16, "Yb": 1.10, "Lu": 0.16, "Hf": 0.43,
    "Rb": 0.6, "Sr": 183, "Th": 0.01, "U": 0.10, "Pb": 0.62,
    "Nb": 0.08, "Ti": 2557, "Zr": 12.5, "Y": 10.6, "Ta": 0.01,
    "Li": nan, "Sc": nan, "V": nan, "Cr": nan, "Ni": 51.6,
    "Na": nan, "K": nan, "Mn": nan, "P": nan, "Co": nan,
    "Ba": 2.88, "Ga": nan, "Cu": nan, "Zn": nan, "Cs": 0.19
}
# fmt: on

# Jordan, Pilet and Brenna - Journal of Petrology (2022)
# fmt: off
PX_JP_2022 = {
    "La": 1.17, "Ce": 3.88, "Pr": nan, "Nd": 4.08, "Sm": 1.60,
    "Eu": 0.62, "Gd": 2.26, "Tb": nan, "Dy": 2.91, "Ho": nan,
    "Er": 1.81, "Tm": nan, "Yb": 1.75, "Lu": 0.26, "Hf": 1.14,
    "Rb": 0.33, "Sr": 45.52, "Th": 0.09, "U": 0.03, "Pb": 0.05,
    "Nb": 1.72, "Ti": 4590, "Zr": 46.19, "Y": 17.57, "Ta": 0.12,
    "Li": nan, "Sc": nan, "V": nan, "Cr": nan, "Ni": nan,
    "Na": nan, "K": nan, "Mn": nan, "P": nan, "Co": nan,
    "Ba": 4.71, "Ga": nan, "Cu": nan, "Zn": nan, "Cs": nan
}
# fmt: on

# Duvernay et al. - ??? (2023)
mnrl_mode_coeff = Dict.empty(key_type=unicode_type, value_type=float64[:])
mnrl_mode_coeff["ol"] = array(
    [
        0.0181671615,
        -0.285097431,
        2.09508358,
        0.0115765041,
        -0.153264241,
        0.519251044,
        0.00149657483,
        -0.0147300510,
        0.572815784,
    ]
)
mnrl_mode_coeff["cpx"] = array(
    [
        -0.25036871,
        2.78713468,
        -9.46264736,
        0.09174549,
        -0.71170887,
        0.69280362,
        -0.0116333,
        0.11198731,
        0.07801724,
    ]
)
mnrl_mode_coeff["pl"] = array(
    [
        0.06651726,
        -0.54734068,
        1.32881577,
        -0.04595634,
        0.42208015,
        -1.09547831,
        0.00857713,
        -0.09025538,
        0.11601566,
    ]
)
mnrl_mode_coeff["spl"] = array(
    [
        0.00701214,
        0.07303097,
        -0.38859093,
        -0.00685826,
        0.03339256,
        0.02103788,
        -0.00759497,
        0.02086458,
        -0.00104847,
    ]
)
mnrl_mode_coeff["grt"] = array(
    [
        0.00179661,
        -0.14451851,
        0.50083179,
        0.01741476,
        -0.10171984,
        -0.09913624,
        -0.00970282,
        0.12271595,
        -0.20676457,
    ]
)
