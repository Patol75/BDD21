from operator import itemgetter
from time import perf_counter

import numpy as np
from data.element_lists import elements_ext_ree
from data.fluidity_simulations import (
    integrated_melting_rate,
    integrated_weighted_concentrations,
)
from data.partition_coefficients import part_coeff, part_coeff_mg
from data.published_compositions import bulk_crust, n_morb_comp, prim_mantle_comp

from bdd21 import MagmaticDifferentiation

chamber_cycle = "rxmtx_mci"
validate = True

part_coeff_key = "white_2014"
part_coeff_mg_key = "laubier_2014"
update_pl_part_coeff = True
excluded_elements = set()

parental_from_fluidity = False
parental_key = "niu_2009"

part_coeff_chamber = part_coeff[part_coeff_key]
if part_coeff_key == "white_2014" and update_pl_part_coeff:
    for element in ["Eu", "Pb"]:
        part_coeff_chamber[element][2] = part_coeff["duvernay_2024"][element][2]
part_coeff_chamber_mg = part_coeff_mg[part_coeff_mg_key]

elements = set(part_coeff_chamber).difference(excluded_elements)
if parental_from_fluidity:
    elements.intersection_update(integrated_weighted_concentrations[parental_key])
else:
    elements.intersection_update(bulk_crust[parental_key])
elements = [ele for ele in elements_ext_ree if ele in elements]
ele_getter = itemgetter(*elements)

if parental_from_fluidity:
    parental_magma = {
        ele: integrated_weighted_concentrations[parental_key][ele]
        / integrated_melting_rate[parental_key]
        for ele in elements
    }
else:
    parental_magma = {
        ele: bulk_crust[parental_key][ele]
        * (prim_mantle_comp["sun_1989"][ele] if parental_key == "niu_2009" else 1.0)
        for ele in elements
    }

magma_diff = MagmaticDifferentiation(
    chamber_cycle,
    part_coeff_chamber,
    part_coeff_chamber_mg,
    parental_magma,
    validate=validate,
    verbose=True,
)

# chamber_parameters = {
#     "mode_ol_gabbro": 0.2,
#     "mode_cpx_gabbro": 0.25,
#     "mgo_repl": 10.0,
#     "mgo_tapp": 7.7,
#     "min_lim_slope": -0.26,
# }

# chamber_parameters = {
#     "prop_troctolite": 0.35,
#     "mode_ol_troctolite": 0.35,
#     "mode_cpx_troctolite": 0.02,
#     "mode_ol_gabbro": 0.11,
#     "mode_cpx_gabbro": 0.3,
#     "mgo_repl": 10.2,
#     "mgo_tapp": 7.8,
#     "min_lim_slope": -0.22,
# }

chamber_parameters = {
    "prop_troctolite": 0.2,
    "mode_ol_troctolite": 0.45,
    "mode_cpx_troctolite": 0.06,
    "mode_ol_gabbro": 0.04,
    "mode_cpx_gabbro": 0.2,
    "prop_ol_tapp_1": 0.31,
    "prop_cpx_tapp_1": 0.0,
    "prop_pl_tapp_1": 0.0,
    "prop_ol_melt_2": 0.04,
    "prop_cpx_melt_2": 0.4,
    "prop_pl_melt_2": 0.4,
    "prop_ol_asml_1": 0.0,
    "prop_cpx_asml_1": 0.0,
    "prop_pl_asml_1": 0.0,
    "prop_ol_asml_2": 0.0,
    "prop_cpx_asml_2": 0.1,
    "prop_pl_asml_2": 0.4,
    "prop_ol_melt_1": 0.0,
    "prop_cpx_melt_1": 0.0,
    "prop_pl_melt_1": 0.0,
    "prop_ol_tapp_2": 0.0,
    "prop_cpx_tapp_2": 0.0,
    "prop_pl_tapp_2": 0.0,
    "mgo_repl": 10.3,
    "mgo_tapp": 7.7,
    "min_lim_slope": -0.2,
}

clock_start = perf_counter()
erupted_product = magma_diff.run(chamber_parameters)
print(f"Elapsed time: {(perf_counter() - clock_start) * 1e3:.3f} ms\n")

n_morb_gale = np.asarray(ele_getter(n_morb_comp["gale_2013"]))
residual = ((erupted_product / n_morb_gale - 1) ** 2).sum() / erupted_product.size

print(f"Residual: {residual:.4f}")
