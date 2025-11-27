import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from numpy import exp, load
from scipy.constants import convert_temperature

# Extract data arrays from zipped archive
npz_file = "../examples/adiabatic_decompression.npz"
data = load(npz_file, allow_pickle=True)
T_pot = data["T_pot"]
adiab_grad = data["adiab_grad"]
elements = data["elements"][()].tolist()
PTF = data["PTF"][()]
mod_abund = data["mod_abund"][()]
conc_sld = data["conc_sld"][()]
conc_lqd = data["conc_lqd"][()]
conc_lqd_cumul = data["conc_lqd_cumul"][()]

# Indices of elements whose concentrations to plot
elements_plot = ["Na", "Ti", "Th", "Ce", "Sm", "Yb"]
elements_ind = [elements.index(ele) for ele in elements_plot]

fig, axes = plt.subplots(
    nrows=2, ncols=6, figsize=(18, 10), constrained_layout=True, sharey=True
)

x_labels = [
    r"Temperature ($^\circ$C)",
    "Melt fraction",
    "Olivine fraction",
    "Orthopyroxene fraction",
    "Clinopyroxene fraction",
    "Aluminous phase fraction",
    *[f"{ele} (ppm)" for ele in elements_plot],
]
panel_labels = map(chr, range(ord("a"), ord("a") + axes.size + 1))

black_text_contour = [
    path_effects.Stroke(linewidth=3, foreground="black"),
    path_effects.Normal(),
]

axes[0, 0].invert_yaxis()
axes[0, 0].set_ylabel("Pressure (GPa)", fontsize=14, fontweight="semibold")
axes[1, 0].set_ylabel("Pressure (GPa)", fontsize=14, fontweight="semibold")

for i, (axis, panel_label, x_label) in enumerate(
    zip(axes.flatten(), panel_labels, x_labels)
):
    axis.grid()

    axis.text(
        0.02 if i in [1, 2, 3] else 0.93,
        0.96,
        panel_label,
        transform=axis.transAxes,
        fontsize=14,
        fontweight="semibold",
        color="white",
        verticalalignment="center",
        horizontalalignment="left",
        path_effects=black_text_contour,
    )

    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel(x_label, fontsize=14, fontweight="semibold")

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    axis.tick_params(which="major", length=7, labelsize=14, width=2)
    axis.tick_params(which="minor", length=4, width=2, color="xkcd:bright red")

axes[0, 0].plot(
    convert_temperature(PTF[1], "K", "C"), PTF[0], linewidth=2, label="Melting path"
)
axes[0, 0].plot(
    convert_temperature(T_pot * exp(adiab_grad / T_pot * PTF[0]), "K", "C"),
    PTF[0],
    linewidth=2,
    linestyle="dashdot",
    color="tab:red",
    label="Adiabatic path",
)
axes[0, 1].plot(PTF[2], PTF[0], linewidth=2)
axes[0, 2].plot(mod_abund[:, 0], PTF[0], linewidth=2)
axes[0, 3].plot(mod_abund[:, 1], PTF[0], linewidth=2)
axes[0, 4].plot(mod_abund[:, 2], PTF[0], linewidth=2)
axes[0, 5].plot(mod_abund[:, 5], PTF[0], linewidth=2, label="Garnet")
axes[0, 5].plot(mod_abund[:, 4], PTF[0], linewidth=2, label="Spinel")
axes[0, 5].plot(mod_abund[:, 3], PTF[0], linewidth=2, label="Plagioclase")

for i, ele_ind in enumerate(elements_ind):
    axes[1, i].plot(
        conc_lqd[:, ele_ind],
        PTF[0],
        linestyle="dotted",
        linewidth=2,
        label="Instantaneous\nliquid melt\ncomposition $c_{l}$",
    )
    axes[1, i].plot(
        conc_lqd_cumul[:, ele_ind],
        PTF[0],
        linewidth=2,
        label="Integrated liquid\ncomposition $C_{l}$",
    )
    axes[1, i].plot(
        conc_sld[:, ele_ind],
        PTF[0],
        linestyle="dashdot",
        linewidth=2,
        label="Residual solid\ncomposition $c_{s}$",
    )

axes[0, 0].legend(
    fontsize=14,
    shadow=True,
    fancybox=True,
    loc="lower left",
    bbox_to_anchor=(0, 0),
    bbox_transform=axes[0, 0].transAxes,
)

axes[0, 5].legend(
    fontsize=14,
    shadow=True,
    fancybox=True,
    loc="center right",
    bbox_to_anchor=(1, 0.65),
    bbox_transform=axes[0, 5].transAxes,
)

leg_comp = axes[1, 2].legend(
    fontsize=14,
    shadow=True,
    fancybox=True,
    loc="center right",
    bbox_to_anchor=(1, 0.65),
    bbox_transform=axes[1, 2].transAxes,
    handlelength=1.8,
)
for leg_han in leg_comp.legend_handles:
    leg_han.set_linewidth(2.5)


fig.savefig("adiabatic_decompression.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
