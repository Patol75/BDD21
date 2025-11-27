from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, concat, read_excel
from scipy.optimize import curve_fit


def func_fit(var, a, b, c, d, e, f, g, h, i):
    x, p = var
    return np.dot(
        [a, b, c, d, e, f, g, h, i],
        [
            x**2 * p**2,
            x**2 * p,
            x**2,
            x * p**2,
            x * p,
            x,
            p**2 * np.ones_like(x),
            p * np.ones_like(x),
            np.ones_like(x),
        ],
    )


def func_fit_cpx_out(var, a, b, c):
    return a * var**2 + b * var + c


def func_fit_cpx_out_katz(var, M_cpx, r0, r1):
    return M_cpx / (r0 + r1 * var)


melt_fraction = np.linspace(0, 1, 101)

data = read_excel("peridotite_experiments.xlsx", usecols=range(1, 10), nrows=57)

trimmed_data = data.drop([15, 20, 34, 56]).copy()
trimmed_data.loc[
    [8, 9, 10, 11, 12, 13, 14, 18, 19, 31, 32, 33, 38, 41, 47, 48, 49, 54, 55], "Cpx"
] = np.nan
trimmed_data.loc[
    np.delete(np.arange(3, 28), np.array([11, 15, 16, 20, 21, 22]) - 3), "Pl"
] = np.nan
trimmed_data.loc[[0, 13, 14, 19], "Spl"] = np.nan
trimmed_data.loc[np.delete(np.arange(28, 56), [34 - 28]), "Spl"] = np.nan
trimmed_data.loc[np.delete(np.arange(56), [15, 20, 34]), "Grt"] = np.nan
print(trimmed_data)

mod_data = trimmed_data.copy()
mod_data = concat(
    [
        mod_data,
        DataFrame(
            [
                [0, 0, 9],
                [0.5, 0, 11],
                [1, 0, 20],
                [1, 17.5, 0],
                [3, 20, 0],
                [4, 24, 0],
                [4.5, 28, 0],
                [5, 0, 32],
                [5, 32, 0],
                [6, 36, 0],
                [7, 41, 0],
            ],
            columns=["P", "F", "Cpx"],
        ),
    ],
    ignore_index=True,
)
mod_data = concat(
    [
        mod_data,
        DataFrame(
            [
                [0, 70, 0],
                [1.5, 0, -10],
                [4, 20, -10],
                [4.5, 25, -10],
                [5, 30, -10],
                [5.5, 35, -10],
                [7, 0, -10],
                [7, 70, 0],
            ],
            columns=["P", "F", "Pl"],
        ),
    ],
    ignore_index=True,
)
mod_data = concat(
    [
        mod_data,
        DataFrame(
            [
                [0, 0, 0],
                [0.5, 0, 0],
                [2.5, 0, 0],
                [5, 0, -10],
                [7, 0, -20],
                [0, 45, 0],
                [1, 50, 0],
                [2, 55, 0],
                [3, 25, 0],
                [3, 55, -1],
                [4, 25, 0],
                [4, 55, -1],
                [5, 50, -3],
                [7, 60, -4],
            ],
            columns=["P", "F", "Spl"],
        ),
    ],
    ignore_index=True,
)
mod_data = concat(
    [
        mod_data,
        DataFrame(
            [
                [0, 75, 0],
                [6, 0, 16],
                [6, 15, 11],
                [6, 30, 5],
                [6, 50, 4],
                [6, 62, 0],
                [7, 0, 17.5],
                [7, 15, 14.3],
                [7, 30, 11.6],
                [7, 50, 6.3],
                [7, 67.43, 0],
                [8, 0, 19],
            ],
            columns=["P", "F", "Grt"],
        ),
    ],
    ignore_index=True,
)
# Garnet exhaustion (Tomlinson and Holland (2021), Fig. 1 Inset)
gnt_exhaust = [
    [2.22, 0, 0],
    [2.40, 10, 0],
    [2.55, 20, 0],
    [3.19, 30, 0],
    [3.70, 40, 0],
    [4.11, 50, 0],
]
mod_data = concat(
    [mod_data, DataFrame(gnt_exhaust, columns=["P", "F", "Grt"])], ignore_index=True
)
# Solidus (Tomlinson and Holland (2021), Fig. 2d)
gnt_solidus = [
    [5.730, 0, 16],
    [4.359, 0, 14],
    [3.463, 0, 12],
    [2.826, 0, 10],
    [2.471, 0, 8],
    [2.390, 0, 6],
    [2.318, 0, 4],
    [2.261, 0, 2],
]
mod_data = concat(
    [mod_data, DataFrame(gnt_solidus, columns=["P", "F", "Grt"])], ignore_index=True
)
# Melting at constant pressure (Tomlinson and Holland (2021), Fig. 7)
gnt_3GPa = [
    [3, 2.08, 9.83],
    [3, 5.21, 8.55],
    [3, 8.39, 7.59],
    [3, 11.51, 6.30],
    [3, 14.69, 5.34],
    [3, 18.02, 4.21],
    [3, 21.15, 3.25],
    [3, 25.83, 0.85],
]
gnt_4GPa = [
    [4, 2.06, 12.80],
    [4, 5.58, 11.84],
    [4, 8.90, 11.04],
    [4, 12.16, 10.24],
    [4, 15.41, 9.44],
    [4, 18.67, 8.72],
    [4, 21.56, 7.92],
    [4, 24.61, 7.20],
    [4, 27.61, 6.24],
    [4, 30.60, 5.28],
    [4, 33.65, 4.32],
    [4, 36.75, 3.36],
    [4, 39.75, 2.40],
    [4, 42.96, 1.76],
    [4, 46.37, 0.96],
    [4, 51.79, 0.32],
]
gnt_5GPa = [
    [5, 2.06, 15.93],
    [5, 6.27, 15.09],
    [5, 10.28, 14.43],
    [5, 14.55, 13.60],
    [5, 19.30, 12.85],
    [5, 23.57, 12.27],
    [5, 28.27, 11.68],
    [5, 33.66, 10.60],
    [5, 39.37, 9.11],
    [5, 45.88, 7.61],
    [5, 51.60, 6.28],
    [5, 59.92, 4.37],
]
mod_data = concat(
    [
        mod_data,
        DataFrame(gnt_3GPa, columns=["P", "F", "Grt"]),
        DataFrame(gnt_4GPa, columns=["P", "F", "Grt"]),
        DataFrame(gnt_5GPa, columns=["P", "F", "Grt"]),
    ],
    ignore_index=True,
)

ol_5GPa = [
    [2.14, 53.85],
    [6.19, 52.18],
    [10.46, 50.52],
    [15.53, 48.52],
    [19.96, 46.86],
    [24.55, 45.03],
    [29.89, 42.95],
    [35.12, 40.87],
    [40.25, 39.04],
    [45.58, 37.05],
    [50.07, 35.38],
    [54.55, 33.72],
    [59.86, 31.81],
]

cpx_1GPa = [
    [2.09, 17.23],
    [5.19, 13.98],
    [9.41, 9.58],
    [12.46, 6.49],
    [15.56, 3.23],
    [17.97, 0.79],
]

min_color = {
    "Ol": "tab:blue",
    "Opx": "tab:orange",
    "Cpx": "tab:green",
    "Pl": "tab:red",
    "Spl": "tab:purple",
    "Grt": "tab:brown",
}

fig, ax = plt.subplots(
    nrows=3,
    ncols=5,
    sharex=True,
    sharey=True,
    constrained_layout=True,
    figsize=(17, 10),
)

# ax[2, 0].scatter(
#     [x / 100 for x, y in ol_5GPa],
#     [y / (100 - x) for x, y in ol_5GPa],
#     s=20,
#     color="xkcd:turquoise",
#     edgecolors="black",
#     linewidths=0.5,
#     zorder=-10,
# )
# ax[0, 2].scatter(
#     [x / 100 for x, y in cpx_1GPa],
#     [y / (100 - x) for x, y in cpx_1GPa],
#     s=20,
#     color="xkcd:turquoise",
#     edgecolors="black",
#     linewidths=0.5,
#     zorder=20,
# )

cpx_out = []
cpx_out_paddy = []
pressure_panel = np.linspace(0, 7, 15)
opx = np.ones((pressure_panel.size, melt_fraction.size))

for mineral in ["Ol", "Cpx", "Pl", "Spl", "Grt"]:
    mask = np.isfinite(mod_data[mineral])
    sigma = np.ones(mask.sum())
    if mineral == "Cpx":
        sigma[[8, 17]] = 100
        sigma[[28, -6, -5]] = 0.1
        sigma[[11, 19, 21, 22, 24, 26]] = 0.05
        sigma[[-11, -10, -9]] = 0.03
        sigma[[4, 14, 18, 20, 23, 27, 32, -8, -7, -4, -3, -2, -1]] = 0.02
        sigma[[30, 31]] = 0.01
    elif mineral == "Grt":
        sigma[
            -np.array([0, 1, 2])
            - len(gnt_5GPa)
            - len(gnt_4GPa)
            - len(gnt_3GPa)
            - len(gnt_solidus)
            - 1
        ] = 0.01
        sigma[
            -np.array([3, 4, 5])
            - len(gnt_5GPa)
            - len(gnt_4GPa)
            - len(gnt_3GPa)
            - len(gnt_solidus)
            - 1
        ] = 0.002

        sigma[
            -np.array([0, 1, 2, 3]) - len(gnt_5GPa) - len(gnt_4GPa) - len(gnt_3GPa) - 1
        ] = 0.01
        sigma[
            -np.array([4, 5, 6, 7]) - len(gnt_5GPa) - len(gnt_4GPa) - len(gnt_3GPa) - 1
        ] = 0.003

        sigma[-np.arange(len(gnt_3GPa) + len(gnt_4GPa)) - len(gnt_5GPa) - 1] = 0.01

        sigma[-np.arange(len(gnt_5GPa)) - 1] = 0.1

        sigma[np.arange(12)] = 0.005
    elif mineral == "Ol":
        sigma[[8, 25]] = 0.3
        sigma[[0, 2, 15, 17]] = 0.18
        sigma[[19, 32, 33, 34, 35, 44, 49, 50, 51]] = 0.1
        sigma[[14, 18, 36, 37, 38, 46, 48, 52]] = 0.06
        sigma[[23, 47]] = 0.05
    elif mineral == "Pl":
        sigma[[-8, -7, -6, -5, -4, -3, -2, -1]] = 0.1
        sigma[[3, 5]] = 0.03
        sigma[[0, 1, 2, 4, 6]] = 0.01
    elif mineral == "Spl":
        sigma[[-1, -2, -3, -4, -5, -6, -7]] = 0.1
        sigma[[-10, -11]] = 0.01
        sigma[[-12, -13, -14]] = 0.001
        sigma[[0, 1, 2, 4, 10, 12, 13, 15, 16]] = 0.001
        sigma[[3, 5, 6, 7, 8, 9, 11, 14, 17, 18, 19, 20, 21]] = 0.01

    popt, pcov = curve_fit(
        func_fit,
        (mod_data["F"][mask] / 100, mod_data["P"][mask]),
        mod_data[mineral][mask] / (100 - mod_data["F"][mask]),
        sigma=sigma,
    )
    print("\n", mineral, popt)

    for i, (axis, pressure) in enumerate(zip(ax.flatten(), pressure_panel)):
        if mineral == "Ol":
            axis.grid()
            axis.tick_params(labelsize=11)
            axis.set_title(f"{pressure} GPa", fontsize=12, fontweight="semibold")

        data_plot = data[abs(data["P"] - pressure) <= 0.2]
        axis.scatter(
            data_plot["F"] / 100,
            data_plot[mineral] / (100 - data_plot["F"]),
            s=40,
            color=min_color[mineral],
            edgecolors="black",
            linewidths=1.25,
            label=mineral,
            zorder=2,
        )
        fit = np.clip(func_fit((melt_fraction, pressure), *popt), 0, 1)
        if mineral == "Grt":
            print("Garnet", fit[0])
        axis.plot(
            melt_fraction,
            fit,
            color=min_color[mineral],
            label="New" if mineral == "Ol" else None,
            linewidth=2,
        )

        opx[i] -= fit
        if mineral == "Cpx":
            cpx_out.append(
                melt_fraction[
                    np.nonzero((fit - 0.000 <= 0) & (melt_fraction > 0.05))[0][0]
                ]
            )

with catch_warnings():
    simplefilter("ignore", category=RuntimeWarning)
    for i, (axis, pressure) in enumerate(zip(ax.flatten(), pressure_panel)):
        data_plot = data[abs(data["P"] - pressure) <= 0.2]
        axis.scatter(
            data_plot["F"] / 100,
            data_plot["Opx"] / (100 - data_plot["F"]),
            s=40,
            color=min_color["Opx"],
            edgecolors="black",
            linewidths=1.25,
            label="Opx",
            zorder=2,
        )
        axis.plot(
            melt_fraction,
            np.clip(opx[i], 0, 1),
            color=min_color["Opx"],
            label="New",
            linewidth=2,
        )

        opx_paddy = np.ones_like(melt_fraction)
        if pressure < 2.5:
            a = -0.115 * pressure**2 + 0.031 * pressure + 0.318
            b = -0.039 * pressure**2 + 0.126 * pressure + 0.419
        else:
            a = 0.048 * pressure**2 + -0.558 * pressure + 1.298
            b = -0.003 * pressure**2 + 0.035 * pressure + 0.445
        ol_paddy = np.clip((a * melt_fraction + b) / (1 - melt_fraction), 0, 1)
        opx_paddy -= ol_paddy
        axis.plot(
            melt_fraction,
            ol_paddy,
            color="tab:blue",
            label="Old",
            linestyle="dashed",
            linewidth=2,
        )

        a = 0.037 * pressure**2 + -0.229 * pressure + -0.606
        b = -0.011 * pressure**2 + 0.112 * pressure + 0.058
        cpx_paddy = np.clip((a * melt_fraction + b) / (1 - melt_fraction), 0, 1)
        opx_paddy -= cpx_paddy
        axis.plot(
            melt_fraction,
            cpx_paddy,
            color="tab:green",
            linestyle="dashed",
            linewidth=2,
        )
        cpx_out_paddy.append(melt_fraction[np.nonzero(cpx_paddy)[0][-1] + 1])
        # cpx_out_paddy.append(
        #     melt_fraction[np.nonzero(cpx_paddy - 0.04 <= 0)[0][0]])

        if pressure <= 2:
            a = 0.026 * pressure**2 + -0.013 * pressure + -0.087
            b = -0.004 * pressure**2 + 0.004 * pressure + 0.02
            spl_paddy = np.clip((a * melt_fraction + b) / (1 - melt_fraction), 0, 1)
            opx_paddy -= spl_paddy
            axis.plot(
                melt_fraction,
                spl_paddy,
                color="tab:purple",
                linestyle="dashed",
                linewidth=2,
            )
            axis.plot(
                melt_fraction,
                np.zeros_like(melt_fraction),
                color="tab:brown",
                linestyle="dashed",
                linewidth=2,
            )
        if pressure >= 2.5:
            a = -0.005 * pressure**2 + 0.078 * pressure + -0.557
            b = -0.001 * pressure**2 + 0.033 * pressure + 0.008
            gnt_paddy = np.clip((a * melt_fraction + b) / (1 - melt_fraction), 0, 1)
            opx_paddy -= gnt_paddy
            axis.plot(
                melt_fraction,
                gnt_paddy,
                color="tab:brown",
                linestyle="dashed",
                linewidth=2,
            )
            axis.plot(
                melt_fraction,
                np.zeros_like(melt_fraction),
                color="tab:purple",
                linestyle="dashed",
                linewidth=2,
            )

        np.clip(opx_paddy, 0, 1, out=opx_paddy)
        axis.plot(
            melt_fraction,
            opx_paddy,
            color="tab:orange",
            linestyle="dashed",
            linewidth=2,
        )

        axis.plot(
            melt_fraction,
            np.zeros_like(melt_fraction),
            color="tab:red",
            linestyle="dashed",
            linewidth=2,
        )

handles, labels = ax[0, 2].get_legend_handles_labels()
unqLab, unqInd = np.unique(labels, return_index=True)
unqHan = np.asarray(handles)[unqInd]
leg_order = [3, 5, 0, 6, 7, 1, 2, 4]
leg = fig.legend(
    unqHan[leg_order],
    unqLab[leg_order],
    ncol=1,
    fontsize=12,
    loc="center right",
    shadow=True,
    fancybox=True,
    bbox_to_anchor=(1, 0.5),
    bbox_transform=ax[1, 2].transAxes,
)


ax[0, 0].set_xlim(-0.025, 0.7)
ax[0, 0].set_ylim(-0.025, 1.025)
ax[-1, 0].set_xlabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[-1, 1].set_xlabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[-1, 2].set_xlabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[-1, 3].set_xlabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[-1, 4].set_xlabel("Melt Fraction", fontsize=12, fontweight="semibold")
ax[0, 0].set_ylabel("Modal Abundance", fontsize=12, fontweight="semibold")
ax[1, 0].set_ylabel("Modal Abundance", fontsize=12, fontweight="semibold")
ax[2, 0].set_ylabel("Modal Abundance", fontsize=12, fontweight="semibold")

fig.savefig("modal_abundance_bdd.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 6))
ax.set_ylim(-0.02, 0.54)
ax.set_xlabel("Pressure (GPa)", fontsize=12, fontweight="semibold")
ax.set_ylabel("Melt Fraction", fontsize=12, fontweight="semibold")

cpx_present_takahashi_1993 = np.array(
    [[0, 0.12], [1.5, 0.17], [4.6, 0.21], [6.5, 0.44]]
)

cpx_present_hirose_1993 = np.array([[1.5, 0.189], [2, 0.219], [2.5, 0.206]])
cpx_exhausted_hirose_1993 = np.array([[1, 0.20]])

ax.scatter(
    pressure_panel,
    cpx_out,
    s=150,
    color="tab:green",
    edgecolors="dimgray",
    linewidths=1.5,
    label="Anchor points",
    zorder=10,
)
ax.scatter(
    pressure_panel,
    cpx_out_paddy,
    s=150,
    color="tab:blue",
    edgecolors="dimgray",
    linewidths=1.5,
    zorder=10,
)

ax.scatter(
    data.query("Cpx > 0")["P"],
    data.query("Cpx > 0")["F"] / 100,
    s=60,
    color="black",
    edgecolors="dimgray",
    linewidths=1.5,
    label="Cpx present",
    zorder=10,
)
ax.scatter(
    data.query("Cpx == 0")["P"],
    data.query("Cpx == 0")["F"] / 100,
    s=60,
    color="white",
    edgecolors="dimgray",
    linewidths=1.5,
    label="Cpx exhausted",
    zorder=10,
)
ax.scatter(
    cpx_present_takahashi_1993[:, 0],
    cpx_present_takahashi_1993[:, 1],
    s=40,
    color="black",
    edgecolors="tab:red",
    linewidths=1.5,
    zorder=10,
)
ax.scatter(
    cpx_present_hirose_1993[:, 0],
    cpx_present_hirose_1993[:, 1],
    s=40,
    color="black",
    edgecolors="tab:red",
    linewidths=1.5,
    zorder=10,
)
ax.scatter(
    cpx_exhausted_hirose_1993[:, 0],
    cpx_exhausted_hirose_1993[:, 1],
    s=40,
    color="white",
    edgecolors="tab:red",
    linewidths=1.5,
    zorder=10,
)

popt, pcov = curve_fit(func_fit_cpx_out, pressure_panel, cpx_out)
print(popt)
pressure_fine = np.linspace(0, 7, 75)
ax.plot(
    pressure_fine,
    func_fit_cpx_out(pressure_fine, *popt),
    color="tab:green",
    linestyle="solid",
    linewidth=3,
    label="New",
)

sigma = np.ones(11)
sigma[[2, 3]] = 0.4
sigma[[1]] = 0.3
popt, pcov = curve_fit(
    func_fit_cpx_out_katz,
    pressure_panel[:11],
    cpx_out_paddy[:11],
    bounds=((0.15, -np.inf, -np.inf), (0.19, np.inf, np.inf)),
    sigma=sigma,
)
print(popt)
ax.plot(
    pressure_fine[:60],
    func_fit_cpx_out_katz(pressure_fine[:60], *popt),
    # 0.17127832 / (0.99133219 - 0.12357699 * pressure_fine[:60]),
    color="tab:blue",
    linestyle="solid",
    linewidth=3,
    label="Old",
)

ax.plot(
    pressure_fine,
    0.17 / (0.5 + 0.08 * pressure_fine),
    color="tab:purple",
    linestyle="solid",
    linewidth=3,
    label="Katz et al. (2003)",
)
# ax.plot(
#     pressure_fine,
#     0.18 / (0.94 + -0.1 * pressure_fine),
#     color="tab:blue",
#     linestyle="solid",
# )

ax.legend(loc="upper left", ncol=1, fontsize=9, shadow=True, fancybox=True)

fig.savefig("exhaustion_cpx.pdf", dpi=300, bbox_inches="tight")
