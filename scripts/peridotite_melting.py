from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_excel
from scipy.optimize import curve_fit


def func_fit(var, a, b, c, d, e, f):
    t, p = var
    temp_solidus = a + b * p + c * p**2
    temp_lherz_liquidus = d + e * p + f * p**2
    return (t - temp_solidus) / (temp_lherz_liquidus - temp_solidus)


def func_fit_2(var, a, b, c):
    t, p = var
    temp_solidus = popt1[0] + popt1[1] * p + popt1[2] * p**2
    temp_lherz_liquidus = popt1[3] + popt1[4] * p + popt1[5] * p**2
    temp_liquidus = a + b * p + c * p**2
    melt_frac_cpx_out = cpx_out[0] * p**2 + cpx_out[1] * p + cpx_out[2]
    temp_cpx_out = temp_solidus + melt_frac_cpx_out ** (1 / beta_1) * (
        temp_lherz_liquidus - temp_solidus
    )
    return (t - temp_cpx_out) / (temp_liquidus - temp_cpx_out)


data = read_excel("peridotite_experiments.xlsx", usecols=range(0, 4))

study_info = [
    # [[0, 2], "", "Borghini et al.\n(2010) — FLZ"],
    [[3, 10], "chartreuse", "Baker & Stolper\n(1994) — MM3"],
    [[11, 20], "limegreen", "Falloon et al.\n(1999) — MM3"],
    [[21, 27], "green", "Falloon et al.\n(2008) — MM3"],
    [[28, 56], "chocolate", "Walter (1998)\nKR4003"],
    [[57, 79], "lightsteelblue", "Takahashi et al.\n(1993) — KLB-1"],
    [[80, 94], "royalblue", "Herzberg et al.\n(2000) — KLB-1"],
    [[95, 107], "cornflowerblue", "Hirose & Kushiro\n(1993) — KLB-1"],
    [[108, 135], "gold", "Kushiro (1996)\nPHN1611"],
    [[136, 143], "pink", "Falloon & Green\n(1988) — MPY-87"],
    [[144, 146], "red", "Lesher et al.\n(2003) — KR4003"],
    [[147, 149], "orangered", "Herzberg & O'Hara\n(2002) — KR4003"],
    [[150, 158], "deeppink", "Robinson et al.\n(1998) — MPY"],
    [[159, 161], "mediumturquoise", "Robinson et al.\n(1998) — TQ"],
    [[162, 176], "blue", "Herzberg & O'Hara\n(2002) — KLB-1"],
    [[177, 189], "darkred", "Tomlinson & Holland\n(2021) — KR4003"],
    [[190, 203], "fuchsia", "Jaques & Green\n(1980) — PY"],
    [[204, 217], "aquamarine", "Jaques & Green\n(1980) — TQ"],
    [[218, 223], "deepskyblue", "Falloon et al.\n(2001) — TQ"],
    [[224, 226], "purple", "Falloon et al.\n(2001) — MPY-87"],
    [[227, 235], "chocolate", "Walter (1998)\nKR4003"],
]

leg_order = {
    0: [3, 0, 1, 2],
    0.5: [3, 0, 2, 5, 4, 1],
    1: [8, 0, 10, 1, 11, 2, 6, 4, 12, 3, 9, 7, 5],
    1.5: [9, 0, 11, 1, 7, 2, 6, 4, 10, 3, 5, 8],
    2: [5, 1, 0, 4, 3, 2],
    2.5: [4, 1, 0, 3, 2],
    3: [0, 5, 2, 1, 4, 3],
    3.5: [3, 2, 0, 1],
    4: [0, 2, 4, 1, 3],
    4.5: [0, 4, 1, 2, 3],
    5: [0, 2, 4, 1, 3],
    5.5: [3, 2, 0, 1],
    6: [0, 3, 1, 2],
    6.5: [3, 0, 1, 2],
    7: [0, 1, 2],
}


beta_1, beta_2 = 1.5, 1.2
cpx_out = [0.00267938, 0.02031577, 0.14623529]

# fmt: off
first_part = [
    3, 4, 5, 6, 7,
    21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 35, 36, 37, 40, 42, 44, 45, 46, 47, 51, 52, 53, 54,
    57, 58, 65, 71, 72, 77, 80, 81, 82, 83, 84, 85,
    144,
    159, 160, 161,
    227, 228, 229, 230, 231, 232, 233, 234, 235,
]
# fmt: on
sigma = np.ones(len(first_part))
sigma[[first_part.index(x) for x in [71]]] = 0.3
sigma[
    [
        first_part.index(x)
        # fmt: off
        for x in [
            3, 4, 5, 6, 7,
            31, 37, 40, 42, 54,
            80, 81, 82, 83, 84, 85,
            227, 228, 229, 230, 231, 232, 233, 234, 235,
        ]
        # fmt: on
    ]
] = 0.1
sigma[[first_part.index(x) for x in [46]]] = 0.05
popt1, pcov1 = curve_fit(
    func_fit,
    (data.loc[first_part, "T"], data.loc[first_part, "P"]),
    (data.loc[first_part, "F"] / 100) ** (1 / beta_1),
    p0=[1085.7, 132.9, -5.1, 1475, 80, -3.2],
    sigma=sigma,
    maxfev=100000,
)
print(popt1)

# fmt: off
second_part = [
    8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 38, 41, 48, 49, 55, 56,
    59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 73, 74, 75, 78, 79,
    162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
]
# fmt: on
sigma = np.ones(len(second_part))
sigma[[second_part.index(x) for x in [73, 78]]] = 0.3
sigma[
    [
        second_part.index(x)
        # fmt: off
        for x in [
            8, 9, 10, 33, 38, 41, 49, 56,
            162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
            177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
        ]
        # fmt: on
    ]
] = 0.1
melt_frac_cpx_out = (
    cpx_out[0] * data.loc[second_part, "P"] ** 2
    + cpx_out[1] * data.loc[second_part, "P"]
    + cpx_out[2]
)
scaled_data = (
    (data.loc[second_part, "F"] / 100 - melt_frac_cpx_out) / (1 - melt_frac_cpx_out)
) ** (1 / beta_2)
popt2, pcov2 = curve_fit(
    func_fit_2,
    (data.loc[second_part, "T"], data.loc[second_part, "P"]),
    scaled_data,
    p0=[1780, 45, -2],
    sigma=sigma,
)
print(popt2)

fig, ax = plt.subplots(
    nrows=3,
    ncols=5,
    sharex=True,
    sharey=True,
    constrained_layout=True,
    figsize=(17, 10),
)

pressure_panel = np.linspace(0, 7, 15)
temperature = np.linspace(1000, 2000, 401)
with catch_warnings():
    simplefilter("ignore", category=RuntimeWarning)
    for axis, pressure in zip(ax.flatten(), pressure_panel):
        axis.set_title(f"{pressure} GPa", fontsize=12, fontweight="semibold")
        axis.grid()
        axis.tick_params(labelsize=11)

        for (lb, ub), study_color, study_label in study_info:
            data_study = data.loc[lb:ub, :]
            data_pres = data_study[abs(data_study["P"] - pressure) <= 0.2]
            if data_pres.empty:
                continue
            axis.scatter(
                data_pres["T"],
                data_pres["F"] / 100,
                s=40,
                color=study_color,
                edgecolors="black",
                linewidths=1.25,
                zorder=2,
                label=study_label,
            )

        # Katz
        temp_solidus = 1085.7 + 132.9 * pressure + -5.1 * pressure**2
        temp_lherz_liquidus = 1475 + 80 * pressure + -3.2 * pressure**2
        temp_liquidus = 1780 + 45 * pressure + -2 * pressure**2
        melt_frac_cpx_out = 0.15 / (0.5 + 0.08 * pressure)
        temp_cpx_out = temp_solidus + melt_frac_cpx_out ** (1 / 1.5) * (
            temp_lherz_liquidus - temp_solidus
        )

        melt_fraction_cpx = (
            (temperature - temp_solidus) / (temp_lherz_liquidus - temp_solidus)
        ) ** 1.5
        melt_fraction_opx = (
            melt_frac_cpx_out
            + (1 - melt_frac_cpx_out)
            * ((temperature - temp_cpx_out) / (temp_liquidus - temp_cpx_out)) ** 1.5
        )

        mask_cpx = temperature <= temp_cpx_out
        mask_opx = temperature >= temp_cpx_out
        axis.plot(
            temperature,
            np.clip(
                np.hstack((melt_fraction_cpx[mask_cpx], melt_fraction_opx[mask_opx])),
                0,
                1,
            ),
            color="tab:purple",
            linewidth=2,
            label="Katz et al.\n(2003)",
        )
        # Paddy
        temp_solidus = 1085.7 + 132.9 * pressure + -5.1 * pressure**2
        temp_lherz_liquidus = 1520 + 80 * pressure + -3.2 * pressure**2
        temp_liquidus = 1780 + 45 * pressure + -2 * pressure**2
        # melt_frac_cpx_out = 0.18 / (0.94 + -0.1 * pressure)
        melt_frac_cpx_out = 0.17127832 / (0.99133219 + -0.12357699 * pressure)
        temp_cpx_out = temp_solidus + melt_frac_cpx_out ** (1 / 1.5) * (
            temp_lherz_liquidus - temp_solidus
        )

        melt_fraction_cpx = (
            (temperature - temp_solidus) / (temp_lherz_liquidus - temp_solidus)
        ) ** 1.5
        melt_fraction_opx = (
            melt_frac_cpx_out
            + (1 - melt_frac_cpx_out)
            * ((temperature - temp_cpx_out) / (temp_liquidus - temp_cpx_out)) ** 1.2
        )
        if pressure == 7:
            melt_fraction_opx[:] = 1

        mask_cpx = temperature <= temp_cpx_out
        mask_opx = temperature >= temp_cpx_out
        axis.plot(
            temperature,
            np.clip(
                np.hstack((melt_fraction_cpx[mask_cpx], melt_fraction_opx[mask_opx])),
                0,
                1,
            ),
            color="tab:blue",
            linewidth=2,
            label="Old",
        )
        # Thomas
        temp_solidus = popt1[:3] @ [1, pressure, pressure**2]
        temp_lherz_liquidus = popt1[3:] @ [1, pressure, pressure**2]
        melt_frac_cpx_out = np.asarray(cpx_out)[::-1] @ [1, pressure, pressure**2]
        temp_cpx_out = temp_solidus + melt_frac_cpx_out ** (1 / beta_1) * (
            temp_lherz_liquidus - temp_solidus
        )

        melt_fraction_cpx = func_fit((temperature, pressure), *popt1) ** beta_1

        melt_fraction_opx = (
            func_fit_2((temperature, pressure), *popt2) ** beta_2
            * (1 - melt_frac_cpx_out)
            + melt_frac_cpx_out
        )

        mask_cpx = temperature <= temp_cpx_out
        mask_opx = temperature >= temp_cpx_out
        axis.plot(
            temperature,
            np.clip(
                np.hstack((melt_fraction_cpx[mask_cpx], melt_fraction_opx[mask_opx])),
                0,
                1,
            ),
            color="tab:green",
            linewidth=2,
            label="New",
        )

        # SORT ITEMS !!!
        handles, labels = axis.get_legend_handles_labels()
        if pressure in [3, 4, 4.5, 5, 6, 7]:
            handles, labels = handles[:-4], labels[:-4]
        else:
            handles, labels = handles[:-3], labels[:-3]
        handles = np.asarray(handles)[leg_order[pressure]]
        labels = np.asarray(labels)[leg_order[pressure]]
        if 1 <= pressure <= 1.5:
            leg = fig.legend(
                handles[1::2],
                labels[1::2],
                ncol=1,
                fontsize=6,
                loc="upper left",
                shadow=True,
                fancybox=True,
                bbox_to_anchor=(0, 1),
                bbox_transform=axis.transAxes,
            )
            leg = fig.legend(
                handles[::2],
                labels[::2],
                ncol=1,
                fontsize=6,
                loc="lower right",
                shadow=True,
                fancybox=True,
                bbox_to_anchor=(1, 0),
                bbox_transform=axis.transAxes,
            )
        else:
            leg = fig.legend(
                handles,
                labels,
                ncol=1,
                fontsize=6,
                loc="upper left",
                shadow=True,
                fancybox=True,
                bbox_to_anchor=(0, 1),
                bbox_transform=axis.transAxes,
            )

for leg_loc in [0, -1]:
    handles, labels = ax[0, leg_loc].get_legend_handles_labels()
    leg = fig.legend(
        handles[-3:],
        labels[-3:],
        ncol=1,
        fontsize=10,
        loc="lower right",
        shadow=True,
        fancybox=True,
        bbox_to_anchor=(1, 0),
        bbox_transform=ax[0, leg_loc].transAxes,
    )
    handles, labels = ax[-1, leg_loc].get_legend_handles_labels()
    leg = fig.legend(
        handles[-3:],
        labels[-3:],
        ncol=1,
        fontsize=10,
        loc="lower left",
        shadow=True,
        fancybox=True,
        bbox_to_anchor=(0, 0),
        bbox_transform=ax[-1, leg_loc].transAxes,
    )

ax[-1, 0].set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="semibold")
ax[-1, 1].set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="semibold")
ax[-1, 2].set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="semibold")
ax[-1, 3].set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="semibold")
ax[-1, 4].set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="semibold")
ax[0, 0].set_ylabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[1, 0].set_ylabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[2, 0].set_ylabel("Melt Fraction", fontsize=12, fontweight="semibold")

fig.savefig("peridotite_melting.pdf", dpi=300, bbox_inches="tight")
