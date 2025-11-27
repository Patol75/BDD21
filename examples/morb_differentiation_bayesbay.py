from operator import itemgetter
from pickle import HIGHEST_PROTOCOL, dump
from time import perf_counter

import numpy as np
from bayesbay import BayesianInversion
from bayesbay.likelihood import LogLikelihood, Target
from bayesbay.parameterization import Parameterization, ParameterSpace
from bayesbay.prior import GaussianPrior, Prior, UniformPrior

from bdd21 import MagmaticDifferentiation


from data.element_lists import elements_ext_ree
from data.fluidity_simulations import (
    integrated_melting_rate,
    integrated_weighted_concentrations,
)
from data.inversion_parameters import initial_parameters, inversion_parameters
from data.partition_coefficients import part_coeff, part_coeff_mg
from data.published_compositions import (
    bulk_crust,
    n_morb_comp,
    n_morb_std,
    prim_mantle_comp,
)


def initialise_parameter(param: Prior, positions: np.ndarray) -> np.ndarray:
    return np.atleast_1d(initial_chamber_parameters[custom_chain_init_key][param.name])


def set_parameter_prior(
    param_name: str, distrib_params: dict[str, float], custom_chain_init: bool = False
) -> Prior:
    if "vmin" in distrib_params:
        parameter = UniformPrior(name=param_name, **distrib_params)
    elif "mean" in distrib_params:
        parameter = GaussianPrior(name=param_name, **distrib_params)
    else:
        raise ValueError(f"Unknown parameter prior used for {param_name}.")

    if custom_chain_init:
        parameter.set_custom_initialize(initialise_parameter)

    return parameter


chamber_cycle = "rxmtx"
validate = True

part_coeff_key = "white_2014"
part_coeff_mg_key = "laubier_2014"
update_pl_part_coeff = True
excluded_elements = set()

gaussian_priors = True
group_parameters = False
invert_parental_magma = True
custom_chain_init = False
custom_chain_init_key = 0
custom_starting_states = True
extensive_sampling = False
chain_count = 104

part_coeff_chamber = part_coeff[part_coeff_key]
if part_coeff_key == "white_2014" and update_pl_part_coeff:
    for element in ["Eu", "Pb"]:
        part_coeff_chamber[element][2] = part_coeff["duvernay_2024"][element][2]
part_coeff_chamber_mg = part_coeff_mg[part_coeff_mg_key]

elements = set(part_coeff_chamber).difference(excluded_elements)
if invert_parental_magma:
    parental_key = "inversion"
else:
    parental_from_fluidity = False
    parental_key = "niu_2009"
    if parental_from_fluidity:
        elements.intersection_update(integrated_weighted_concentrations[parental_key])
    else:
        elements.intersection_update(bulk_crust[parental_key])
elements = [ele for ele in elements_ext_ree if ele in elements]
ele_getter = itemgetter(*elements)

if invert_parental_magma:
    parental_magma = {}
elif parental_from_fluidity:
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
)

try:
    chamber_parameters = inversion_parameters[chamber_cycle]
except KeyError:
    chamber_parameters = inversion_parameters["single_cryst_no_mci"]
initial_chamber_parameters = initial_parameters[chamber_cycle]

if gaussian_priors:
    chamber_parameters["mgo_tapp"] = {"mean": 7.76, "std": 0.045, "perturb_std": 4e-3}

param_spaces = []
if group_parameters:
    param_groups = {}
    param_groups["mgo"] = ["mgo_repl", "mgo_tapp", "min_lim_slope"]
    if chamber_cycle.count("x") >= 1:
        param_groups["modes_gabbro"] = ["mode_ol_gabbro", "mode_cpx_gabbro"]
    if chamber_cycle.count("x") == 2:
        param_groups["modes_troctolite"] = [
            "prop_troctolite",
            "mode_ol_troctolite",
            "mode_cpx_troctolite",
        ]
    if "mci" in chamber_cycle:
        for mnrl in ["ol", "cpx", "pl"]:
            param_groups[f"{mnrl}_mci_event_1"] = [
                f"prop_{mnrl}_asml_1",
                f"prop_{mnrl}_melt_1",
                f"prop_{mnrl}_tapp_1",
            ]
            param_groups[f"{mnrl}_mci_event_2"] = [
                f"prop_{mnrl}_asml_2",
                f"prop_{mnrl}_melt_2",
                f"prop_{mnrl}_tapp_2",
            ]

    for param_group_name, param_group in param_groups.items():
        parameters = []
        for param_name in param_group:
            distrib_params = chamber_parameters.pop(param_name)
            parameter = set_parameter_prior(param_name, distrib_params)
            parameters.append(parameter)

        param_spaces.append(
            ParameterSpace(name=param_group_name, n_dimensions=1, parameters=parameters)
        )

for param_name, distrib_params in chamber_parameters.items():
    parameter = set_parameter_prior(param_name, distrib_params)

    param_spaces.append(
        ParameterSpace(name=param_name, n_dimensions=1, parameters=[parameter])
    )

if invert_parental_magma:
    magma_params = {}
    index_Be = elements_ext_ree.index("Be")
    index_Gd = elements_ext_ree.index("Gd")
    for ele in elements:
        if validate:
            index_ele = elements_ext_ree.index(ele)
            if ele == "Sr" or ele == "Sc":
                min_enrichment_factor = 1.0
            elif index_ele < index_Be:
                min_enrichment_factor = 1.8
            elif index_ele < index_Gd:
                min_enrichment_factor = 1.4
            else:
                min_enrichment_factor = 1.2
        else:
            min_enrichment_factor = 1.0

        conc = n_morb_comp["gale_2013"][ele]
        magma_params[ele] = {
            "vmin": conc / 10.0,
            "vmax": conc / min_enrichment_factor,
            "perturb_std": 5e-3 * conc,
        }
        if custom_chain_init:
            initial_chamber_parameters[custom_chain_init_key][ele] = conc / 3.0

    parameters = []
    for param_name, distrib_params in magma_params.items():
        parameter = set_parameter_prior(param_name, distrib_params, custom_chain_init)
        parameters.append(parameter)

    param_spaces.append(
        ParameterSpace(name="parental_magma", n_dimensions=1, parameters=parameters)
    )

parameterisation = Parameterization(param_spaces)

if custom_starting_states:
    walkers_starting_states = []
    init_param_sets_count = len(initial_chamber_parameters)

    for chain_index in range(chain_count):
        starting_state = parameterisation.initialize()
        dict_key = chain_index % init_param_sets_count
        for space_name, space_state in starting_state.param_values.items():
            if space_name == "parental_magma":
                continue
            for p_name in space_state.param_values:
                space_state.set_param_values(
                    p_name, np.atleast_1d(initial_chamber_parameters[dict_key][p_name])
                )

        walkers_starting_states.append(starting_state)
else:
    walkers_starting_states = None

n_morb_gale = np.asarray(ele_getter(n_morb_comp["gale_2013"]))
n_morb_std_gale = np.asarray(ele_getter(n_morb_std["gale_2013"])) / 2.0
covariance_mat_inv = np.diag(n_morb_std_gale**-2.0)
target = Target("n_morb", n_morb_gale, covariance_mat_inv=covariance_mat_inv)

log_likelihood = LogLikelihood(targets=target, fwd_functions=magma_diff.run)

inversion = BayesianInversion(
    parameterization=parameterisation,
    log_likelihood=log_likelihood,
    walkers_starting_states=walkers_starting_states,
    n_chains=chain_count,
)

clock_start = perf_counter()
if extensive_sampling:
    inversion.run(
        n_iterations=2_000_000,
        burnin_iterations=250_000,
        save_every=1_000,
        verbose=False,
    )
else:
    inversion.run(
        n_iterations=200_000, burnin_iterations=50_000, save_every=100, verbose=False
    )
elapsed_time = perf_counter() - clock_start
print(f"Elapsed time: {int(elapsed_time / 60):02d}:{elapsed_time % 60:.3f}")

inversion.chains[0].print_statistics()
results = inversion.get_results()

pickled_results = {}
pickled_results["results"] = results
pickled_results["elements"] = elements
pickled_results["parental_magma"] = parental_magma

validate_suffix = "_validate" if validate else ""
extensive_suffix = "_extensive" if extensive_sampling else ""
group_suffix = "_group" if group_parameters else ""
pickle_name = "_".join([chamber_cycle, parental_key, part_coeff_key, part_coeff_mg_key])
pickle_name += validate_suffix + extensive_suffix + group_suffix
with open(f"inversion_results_08_oct/{pickle_name}.pickle", "wb") as pkl_file:
    dump(pickled_results, pkl_file, HIGHEST_PROTOCOL)
