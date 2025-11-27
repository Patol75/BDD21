from operator import itemgetter
from pickle import load

import arviz as az
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from data.fluidity_simulations import (
    integrated_melting_rate,
    integrated_weighted_concentrations,
)
from data.published_compositions import (
    bulk_crust,
    n_morb_comp,
    n_morb_std,
    prim_mantle_comp,
)


def get_param_key(param_name: str) -> str:
    for results_key in results:
        if param_name in results_key:
            return results_key


def plot_fill_between(axis, compositions, normalisation=1.0, color=None, labels=None):
    labels = labels or [None, None]

    axis.plot(
        elements,
        compositions[1] / normalisation,
        linestyle="dashdot",
        color=color,
        label=labels[0],
    )
    axis.fill_between(
        elements,
        compositions[0] / normalisation,
        compositions[2] / normalisation,
        alpha=0.4,
        color=color,
        label=labels[1],
    )


def plot_composition_dictionary(
    axis, composition, normalisation=None, interval=None, color=None, label=None
):
    local_ele_map = {i: ele for i, ele in enumerate(elements) if ele in composition}
    indices = local_ele_map.keys()
    local_elements = local_ele_map.values()

    if normalisation is None:
        concentrations = [composition[ele] for ele in local_elements]
    else:
        concentrations = [
            composition[ele] / normalisation[ele] for ele in local_elements
        ]

    if interval is None:
        axis.plot(
            indices,
            concentrations,
            label=label,
            linestyle="none",
            marker="X",
            markeredgecolor="black",
            markerfacecolor=color,
            markersize=10,
        )
    else:
        lower_limit = [interval[0][ele] for ele in local_elements]
        upper_limit = [interval[1][ele] for ele in local_elements]
        confidence_interval = np.vstack((lower_limit, upper_limit))

        axis.errorbar(indices, concentrations, yerr=confidence_interval)


chamber_cycle = "rxmtx"
parental_key = "inversion"
# part_coeff_key = "claeson_2004"
# part_coeff_key = "gao_2007"
# part_coeff_key = "o_neill_2012"
# part_coeff_key = "laubier_2014"
part_coeff_key = "white_2014"
# part_coeff_key = "shorttle_2016"
# part_coeff_key = "jordan_2022"
part_coeff_mg_key = "laubier_2014"
# part_coeff_mg_key = "o_neill_2012"
# suffix = ""
# suffix = "_extensive"
suffix = "_validate"
# suffix = "_validate_extensive"
# suffix = "_validate_extensive_group"
pickle_name = "_".join([chamber_cycle, parental_key, part_coeff_key, part_coeff_mg_key])
pickle_name += suffix
plot_subset = False
subset_suffix = "_subset" if plot_subset else ""
pdf_name = pickle_name + subset_suffix

with open(
    f"../examples/inversion_results_08_oct/{pickle_name}.pickle", "rb"
) as pkl_file:
    pickled_results = load(pkl_file)
results = pickled_results["results"]
elements = pickled_results["elements"]
parental_magma = pickled_results["parental_magma"]

element_count = len(elements)
ele_getter = itemgetter(*elements)

normalisation_niu_2009 = {
    ele: prim_mantle_comp["mcdonough_1995"][ele] / prim_mantle_comp["sun_1989"][ele]
    for ele in prim_mantle_comp["sun_1989"]
}
fluidity_parental_magma = {
    ele: value / integrated_melting_rate["d9_wh"]
    for ele, value in integrated_weighted_concentrations["d9_wh"].items()
}

prim_mant_ms = np.asarray(ele_getter(prim_mantle_comp["mcdonough_1995"]))
inferred_conc = np.asarray(results[get_param_key("dpred")])
erupted_product = inferred_conc / prim_mant_ms
erupted_product_pctl = np.percentile(erupted_product, (10, 50, 90), axis=0)

n_morb_gale = np.asarray(ele_getter(n_morb_comp["gale_2013"]))
n_morb_std_gale = np.asarray(ele_getter(n_morb_std["gale_2013"]))
residual = np.sum((inferred_conc / n_morb_gale - 1.0) ** 2, axis=1) / element_count
residual_pctl = np.percentile(residual, (10, 50, 90))

mode_mci_label = ", before MCI)" if "mci" in chamber_cycle else ")"
x_labels = {
    "frac_cryst": "Fraction crystallised",
    "frac_tapp": "Fraction erupted",
    "frac_sum": "Sum of cycle fractions",
    "frac_ratio": "Ratio of crystallised to tapped fractions",
    "mode_ol_gabbro": "Ol mode (layer 3)",
    "mode_cpx_gabbro": "Cpx mode (layer 3)",
    "mode_pl_gabbro": "Pl mode (layer 3)",
    "mgo_repl": "MgO concentration (wt.%, replenishing)",
    "mgo_tapp": "MgO concentration (wt.%, erupted)",
    "min_lim_slope": "Minimum limiting slope",
    "concentrations": "Trace element",
    "residual": "Residual (inference versus N-MORB)",
}
if chamber_cycle.count("x") == 2:
    x_labels |= {
        "prop_troctolite": "Proportion of troctolites",
        "mode_ol_troctolite": f"Ol mode (troctolite{mode_mci_label}",
        "mode_cpx_troctolite": f"Cpx mode (troctolite{mode_mci_label}",
        "mode_pl_troctolite": f"Pl mode (troctolite{mode_mci_label}",
        "mode_ol_gabbro": f"Ol mode (gabbro{mode_mci_label}",
        "mode_cpx_gabbro": f"Cpx mode (gabbro{mode_mci_label}",
        "mode_pl_gabbro": f"Pl mode (gabbro{mode_mci_label}",
        "mode_ol_layer_3": "Ol mode (layer 3)",
        "mode_cpx_layer_3": "Cpx mode (layer 3)",
        "mode_pl_layer_3": "Pl mode (layer 3)",
    }
if "mci" in chamber_cycle:
    x_labels |= {
        "prop_ol_asml_1": "Ol crystals assimilated (troctolite)",
        "prop_cpx_asml_1": "Cpx crystals assimilated (troctolite)",
        "prop_pl_asml_1": "Pl crystals assimilated (troctolite)",
        "prop_ol_melt_1": "Ol crystals remelted (troctolite)",
        "prop_cpx_melt_1": "Cpx crystals remelted (troctolite)",
        "prop_pl_melt_1": "Pl crystals remelted (troctolite)",
        "prop_ol_tapp_1": "Ol crystals erupted (troctolite)",
        "prop_cpx_tapp_1": "Cpx crystals erupted (troctolite)",
        "prop_pl_tapp_1": "Pl crystals erupted (troctolite)",
        "prop_ol_asml_2": "Ol crystals assimilated (gabbro)",
        "prop_cpx_asml_2": "Cpx crystals assimilated (gabbro)",
        "prop_pl_asml_2": "Pl crystals assimilated (gabbro)",
        "prop_ol_melt_2": "Ol crystals remelted (gabbro)",
        "prop_cpx_melt_2": "Cpx crystals remelted (gabbro)",
        "prop_pl_melt_2": "Pl crystals remelted (gabbro)",
        "prop_ol_tapp_2": "Ol crystals erupted (gabbro)",
        "prop_cpx_tapp_2": "Cpx crystals erupted (gabbro)",
        "prop_pl_tapp_2": "Pl crystals erupted (gabbro)",
        "mode_ol_troctolite_final": "Ol mode (troctolite, after MCI)",
        "mode_cpx_troctolite_final": "Cpx mode (troctolite, after MCI)",
        "mode_pl_troctolite_final": "Pl mode (troctolite, after MCI)",
        "mode_ol_gabbro_final": "Ol mode (gabbro, after MCI)",
        "mode_cpx_gabbro_final": "Cpx mode (gabbro, after MCI)",
        "mode_pl_gabbro_final": "Pl mode (gabbro, after MCI)",
    }

text_contour = [
    path_effects.Stroke(linewidth=2, foreground="black"),
    path_effects.Normal(),
]

hist_kwargs = {"density": True, "alpha": 0.8, "edgecolor": "gray", "linewidth": 0.5}
hist_subset_kwargs = hist_kwargs | {"alpha": 0.5, "facecolor": "fuchsia"}

parameters = {}
for param_name in x_labels:
    param_key = get_param_key(param_name)
    if param_key is not None:
        parameters[param_name] = np.ravel(results[param_key])
parameters["mode_pl_gabbro"] = (
    1.0 - parameters["mode_ol_gabbro"] - parameters["mode_cpx_gabbro"]
)
if chamber_cycle.count("x") == 2:
    parameters["mode_pl_troctolite"] = (
        1.0 - parameters["mode_ol_troctolite"] - parameters["mode_cpx_troctolite"]
    )
parameters["frac_sum"] = parameters["frac_cryst"] + parameters["frac_tapp"]
parameters["frac_ratio"] = parameters["frac_cryst"] / parameters["frac_tapp"]

if plot_subset:
    subset_ind = np.flatnonzero(
        (residual <= 1e-3)
        & (parameters["frac_ratio"] >= 2.1)
        & (parameters["frac_ratio"] <= 2.6)
        & (parameters["mgo_repl"] >= 10.0)
        & (parameters["min_lim_slope"] <= -0.19)
        & (parameters["prop_troctolite"] >= 0.1)
        & (parameters["mode_ol_gabbro"] <= 0.2)
        & (parameters["mode_pl_gabbro"] <= 0.7)
        & (parameters["prop_cpx_melt_2"] > parameters["prop_ol_melt_2"])
        & (parameters["prop_pl_melt_2"] > parameters["prop_ol_melt_2"])
    )

    parameters_subset = {
        param_name: param_value[subset_ind]
        for param_name, param_value in parameters.items()
    }

    ind_smallest_res = np.argpartition(residual[subset_ind], 3)[:3]
    for param_name, param_array in parameters.items():
        print(param_name, parameters_subset[param_name][ind_smallest_res])
    print(residual[subset_ind][ind_smallest_res])

    chamber_parameters = [
        "prop_troctolite",
        "mode_ol_troctolite",
        "mode_cpx_troctolite",
        "mode_ol_gabbro",
        "mode_cpx_gabbro",
        "mgo_repl",
        "mgo_tapp",
        "min_lim_slope",
        "prop_ol_tapp_1",
        "prop_cpx_tapp_1",
        "prop_pl_tapp_1",
        "prop_ol_melt_2",
        "prop_cpx_melt_2",
        "prop_pl_melt_2",
    ]
    best_fit_params = {
        param_name: parameters_subset[param_name][ind_smallest_res][0]
        for param_name in chamber_parameters
    }
else:
    parameters_subset = {param_name: [] for param_name in parameters}
    subset_ind = []

fig = plt.figure(figsize=(20, 10), layout="constrained")

mci_mode = "_final" if "mci" in chamber_cycle else ""
if chamber_cycle.count("x") == 1:
    gs = GridSpec(6, 12, figure=fig)

    axes = {}
    axes["frac_sum"] = fig.add_subplot(gs[:2, :3])
    axes["frac_ratio"] = fig.add_subplot(gs[:2, 3:6])
    axes["mode_ol_gabbro"] = fig.add_subplot(gs[2:4, :2])
    axes["mode_cpx_gabbro"] = fig.add_subplot(gs[2:4, 2:4])
    axes["mode_pl_gabbro"] = fig.add_subplot(gs[2:4, 4:6])
    axes["mgo_repl"] = fig.add_subplot(gs[4:, :2])
    axes["mgo_tapp"] = fig.add_subplot(gs[4:, 2:4])
    axes["min_lim_slope"] = fig.add_subplot(gs[4:, 4:6])
    axes["concentrations"] = fig.add_subplot(gs[:3, 6:])
    axes["residual"] = fig.add_subplot(gs[3:, 6:])
else:
    gs = GridSpec(8, 12, figure=fig)

    axes = {}
    axes["frac_sum"] = fig.add_subplot(gs[:2, :2])
    axes["frac_ratio"] = fig.add_subplot(gs[:2, 2:4])
    axes["prop_troctolite"] = fig.add_subplot(gs[:2, 4:6])
    axes[f"mode_ol_troctolite{mci_mode}"] = fig.add_subplot(gs[2:4, :2])
    axes[f"mode_cpx_troctolite{mci_mode}"] = fig.add_subplot(gs[2:4, 2:4])
    axes[f"mode_pl_troctolite{mci_mode}"] = fig.add_subplot(gs[2:4, 4:6])
    axes[f"mode_ol_gabbro{mci_mode}"] = fig.add_subplot(gs[4:6, :2])
    axes[f"mode_cpx_gabbro{mci_mode}"] = fig.add_subplot(gs[4:6, 2:4])
    axes[f"mode_pl_gabbro{mci_mode}"] = fig.add_subplot(gs[4:6, 4:6])
    axes["mgo_repl"] = fig.add_subplot(gs[6:, :2])
    axes["mgo_tapp"] = fig.add_subplot(gs[6:, 2:4])
    axes["min_lim_slope"] = fig.add_subplot(gs[6:, 4:6])
    axes["concentrations"] = fig.add_subplot(gs[:4, 6:])
    axes["mode_ol_layer_3"] = fig.add_subplot(gs[4:6, 6:8])
    axes["mode_cpx_layer_3"] = fig.add_subplot(gs[4:6, 8:10])
    axes["mode_pl_layer_3"] = fig.add_subplot(gs[4:6, 10:])
    axes["residual"] = fig.add_subplot(gs[6:, 6:])

    for suffix in [f"troctolite{mci_mode}", "layer_3"]:
        axes[f"mode_ol_{suffix}"].set_ylabel("Density", fontsize=12)

for ax_name, ax in axes.items():
    ax.tick_params(labelsize=12)
    ax.set_xlabel(x_labels[ax_name], fontsize=12)

for ax_name in ["frac_sum", f"mode_ol_gabbro{mci_mode}", "mgo_repl", "residual"]:
    axes[ax_name].set_ylabel("Density", fontsize=12)
axes["concentrations"].set_ylabel("Normalised concentration", fontsize=12)

for ax_name, ax in axes.items():
    if ax_name in parameters:
        if ax_name.startswith("frac"):
            facecolor = "chartreuse"
        elif ax_name.startswith("mode") or ax_name.startswith("prop"):
            facecolor = "darkviolet"
        else:
            facecolor = "cornflowerblue"

        ax.hist(parameters[ax_name], bins=30, facecolor=facecolor, **hist_kwargs)
        ax.hist(parameters_subset[ax_name], bins=30, **hist_subset_kwargs)

axes[f"mode_pl_gabbro{mci_mode}"].hist(
    parameters[f"mode_pl_gabbro{mci_mode}"],
    bins=30,
    facecolor="darkviolet",
    **hist_kwargs,
)
axes[f"mode_pl_gabbro{mci_mode}"].hist(
    parameters_subset[f"mode_pl_gabbro{mci_mode}"], bins=30, **hist_subset_kwargs
)

axes["residual"].hist(residual, bins=80, facecolor="turquoise", **hist_kwargs)
axes["residual"].hist(residual[subset_ind], bins=20, **hist_subset_kwargs)
axes["residual"].text(
    0.99,
    0.98,
    f"10th percentile: {residual_pctl[0]:.3e}\n"
    f"median: {residual_pctl[1]:.3e}\n"
    f"90th percentile: {residual_pctl[2]:.3e}",
    color="white",
    fontsize=12,
    fontweight="semibold",
    horizontalalignment="right",
    path_effects=text_contour,
    transform=axes["residual"].transAxes,
    verticalalignment="top",
)

plot_fill_between(
    axes["concentrations"],
    erupted_product_pctl,
    color="crimson",
    labels=["Inferred basalt (median)", "Inferred basalt (10th-90th percentiles)"],
)
plot_fill_between(
    axes["concentrations"],
    [n_morb_gale - n_morb_std_gale, n_morb_gale, n_morb_gale + n_morb_std_gale],
    normalisation=prim_mant_ms,
    color="gray",
    labels=["N-MORB (Gale et al., 2013)", "N-MORB 95% confidence (Gale et al., 2013)"],
)

if any("parental_magma" in results_key for results_key in results):
    parental_magma_inferred_pctl = np.empty((3, element_count))
    for i, ele in enumerate(elements):
        ele_conc = np.ravel(results[f"parental_magma.{ele}"]) / prim_mant_ms[i]
        parental_magma_inferred_pctl[:, i] = np.percentile(ele_conc, (10, 50, 90))
        print(
            ele, n_morb_gale[i] / prim_mant_ms[i] / parental_magma_inferred_pctl[1, i]
        )

    plot_fill_between(
        axes["concentrations"],
        parental_magma_inferred_pctl,
        color="gold",
        labels=[
            "Inferred parental magma (median)",
            "Inferred parental magma (10th-90th percentiles)",
        ],
    )

plot_composition_dictionary(
    axes["concentrations"],
    bulk_crust["niu_2009"],
    normalisation=normalisation_niu_2009,
    color="fuchsia",
    label="Bulk crust (Niu and O'Hara, 2009)",
)
plot_composition_dictionary(
    axes["concentrations"],
    fluidity_parental_magma,
    normalisation=prim_mantle_comp["mcdonough_1995"],
    color="darkorange",
    label="Modelled parental magma (Duvernay et al., 2024)",
)

axes["concentrations"].set_xticks(np.arange(element_count), elements)
axes["concentrations"].set_yscale("log")
axes["concentrations"].grid(which="both")
leg = axes["concentrations"].legend(
    loc="lower right", bbox_to_anchor=[1.0, 0.0], fontsize=12
)

fig.savefig(f"figures_08_oct/posterior_{pdf_name}.pdf", dpi=300)
plt.close(fig)

if "mci" in chamber_cycle:
    fig = plt.figure(figsize=(20, 10), layout="constrained")

    gs = GridSpec(5, 6, figure=fig)

    axes = {}
    axes["mode_ol_troctolite"] = fig.add_subplot(gs[0, 0])
    axes["mode_cpx_troctolite"] = fig.add_subplot(gs[0, 1])
    axes["mode_pl_troctolite"] = fig.add_subplot(gs[0, 2])
    axes["mode_ol_gabbro"] = fig.add_subplot(gs[0, 3])
    axes["mode_cpx_gabbro"] = fig.add_subplot(gs[0, 4])
    axes["mode_pl_gabbro"] = fig.add_subplot(gs[0, 5])
    axes["prop_ol_asml_1"] = fig.add_subplot(gs[1, 0])
    axes["prop_cpx_asml_1"] = fig.add_subplot(gs[1, 1])
    axes["prop_pl_asml_1"] = fig.add_subplot(gs[1, 2])
    axes["prop_ol_asml_2"] = fig.add_subplot(gs[1, 3])
    axes["prop_cpx_asml_2"] = fig.add_subplot(gs[1, 4])
    axes["prop_pl_asml_2"] = fig.add_subplot(gs[1, 5])
    axes["prop_ol_melt_1"] = fig.add_subplot(gs[2, 0])
    axes["prop_cpx_melt_1"] = fig.add_subplot(gs[2, 1])
    axes["prop_pl_melt_1"] = fig.add_subplot(gs[2, 2])
    axes["prop_ol_melt_2"] = fig.add_subplot(gs[2, 3])
    axes["prop_cpx_melt_2"] = fig.add_subplot(gs[2, 4])
    axes["prop_pl_melt_2"] = fig.add_subplot(gs[2, 5])
    axes["prop_ol_tapp_1"] = fig.add_subplot(gs[3, 0])
    axes["prop_cpx_tapp_1"] = fig.add_subplot(gs[3, 1])
    axes["prop_pl_tapp_1"] = fig.add_subplot(gs[3, 2])
    axes["prop_ol_tapp_2"] = fig.add_subplot(gs[3, 3])
    axes["prop_cpx_tapp_2"] = fig.add_subplot(gs[3, 4])
    axes["prop_pl_tapp_2"] = fig.add_subplot(gs[3, 5])
    axes["mode_ol_troctolite_final"] = fig.add_subplot(gs[4, 0])
    axes["mode_cpx_troctolite_final"] = fig.add_subplot(gs[4, 1])
    axes["mode_pl_troctolite_final"] = fig.add_subplot(gs[4, 2])
    axes["mode_ol_gabbro_final"] = fig.add_subplot(gs[4, 3])
    axes["mode_cpx_gabbro_final"] = fig.add_subplot(gs[4, 4])
    axes["mode_pl_gabbro_final"] = fig.add_subplot(gs[4, 5])

    for ax_name, ax in axes.items():
        ax.tick_params(labelsize=12)
        ax.set_xlabel(x_labels[ax_name], fontsize=12)

        if "_ol_" in ax_name:
            axes[ax_name].set_ylabel("Density", fontsize=12)

    for ax_name, ax in axes.items():
        if ax_name in parameters:
            if ax_name.startswith("prop"):
                facecolor = "orange"
            elif ax_name.endswith("final"):
                facecolor = "red"
            else:
                facecolor = "yellow"
            ax.hist(parameters[ax_name], bins=30, facecolor=facecolor, **hist_kwargs)
            ax.hist(parameters_subset[ax_name], bins=30, **hist_subset_kwargs)

    fig.savefig(f"figures_08_oct/posterior_modes_{pdf_name}.pdf", dpi=300)
    plt.close(fig)

    az.rcParams["plot.max_subplots"] = 1000
    pair_plot = az.plot_pair(
        parameters_subset if plot_subset else parameters,
        var_names=["~frac", "~mode_pl"],  # , "~prop_ol", "~prop_cpx", "~prop_pl"],
        filter_vars="like",
        marginals=True,
        figsize=(60, 60),
        textsize=16,
        kind="kde",
        kde_kwargs={
            "hdi_probs": [0.3, 0.6, 0.9],
            "contourf_kwargs": {"cmap": "plasma"},
        },
        backend_kwargs={"layout": "constrained"},
        marginal_kwargs={"plot_kwargs": {"linewidth": 2}},
        reference_values=best_fit_params if plot_subset else None,
        reference_values_kwargs={
            "markeredgecolor": "black",
            "markerfacecolor": "dodgerblue",
            "markersize": 12,
        },
    )

    az_fig = pair_plot.ravel()[0].figure
    az_fig.savefig(f"figures_08_oct/posterior_pair_plot_{pdf_name}.pdf", dpi=300)
    plt.close(az_fig)
