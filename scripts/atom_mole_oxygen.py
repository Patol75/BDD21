from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from pandas import read_excel
from scipy.optimize import curve_fit


def compo_per_oxygen(oxy, compo, mask, molar_mass, oxy_per_oxide, cat_per_oxide):
    oxy_numb = compo[:, 3:] / molar_mass[mask] * oxy_per_oxide[mask]
    return (
        oxy_numb
        * oxy
        / np.tile(oxy_numb.sum(axis=1), (oxy_numb.shape[1], 1)).T
        * cat_per_oxide[mask]
        / oxy_per_oxide[mask]
    )


def linear_fit(var, a, b, c, d):
    x, t = var
    return a * x * t + b * x + c * t + d


def label_match(label):
    match label:
        case "Melt Fraction":
            return 2
        case "Temperature (K)":
            return 1
        case "Pressure (GPa)":
            return 0
        case _:
            raise AttributeError


def fit_and_plot(axis, mask, xy_data, z_data, fit_label, surf_color):
    if mask.sum() == 0:
        return

    melt_frac = np.linspace(0, xy_data[mask, 2].max(), 100)
    temp = np.linspace(xy_data[mask, 1].min(), xy_data[mask, 1].max(), 100)
    melt_grid, temp_grid = np.meshgrid(melt_frac, temp)

    mask_zero = z_data[mask] == 0
    sigma = np.ones_like(z_data[mask])
    sigma[mask_zero] = np.inf
    popt, pcov = curve_fit(
        linear_fit, (xy_data[mask, 2], xy_data[mask, 1]), z_data[mask], sigma=sigma
    )
    print(fit_label, *(f"{val:.3e}" for val in popt))

    try:
        assert_allclose(
            linear_fit(
                (xy_data[mask, 2][~mask_zero], xy_data[mask, 1][~mask_zero]), *popt
            ),
            z_data[mask][~mask_zero],
            rtol=0.3,
            atol=0,
        )
    except AssertionError as assert_error:
        print(assert_error.args[0])

    axis.plot_surface(
        melt_grid,
        temp_grid,
        linear_fit((melt_grid, temp_grid), *popt),
        color=surf_color,
        alpha=0.5,
    )


def plot_panel_3d(
    axis,
    z_label,
    mask,
    xy_data,
    z_data,
    x_label="Melt Fraction",
    y_label="Temperature (K)",
):
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_zlabel(z_label)

    x_data_col = label_match(x_label)
    y_data_col = label_match(y_label)

    axis.plot(
        xy_data[mask, x_data_col],
        xy_data[mask, y_data_col],
        z_data[mask],
        linestyle="none",
        marker="o",
        color="blue",
    )
    axis.plot(
        xy_data[~mask, x_data_col],
        xy_data[~mask, y_data_col],
        z_data[~mask],
        linestyle="none",
        marker="o",
        color="green",
    )

    fit_and_plot(axis, mask, xy_data, z_data, z_label + " Spinel", "blue")
    fit_and_plot(axis, ~mask, xy_data, z_data, z_label + " Garnet", "green")


col_range = list(
    chain(
        range(1, 4),
        range(12, 22),
        range(23, 30),
        range(31, 40),
        range(41, 50),
        range(51, 60),
    )
)
data = read_excel(
    "peridotite_experiments.xlsx",
    usecols=col_range,
    skiprows=[1, 2, 3, 16, 35],
    nrows=53,
)

data["T"] += 273.15
data["F"] /= 100

melt_compo = (
    data[
        [
            "P",
            "T",
            "F",
            "SiO2",
            "TiO2",
            "Cr2O3",
            "Al2O3",
            "FeO",
            "MgO",
            "CaO",
            "MnO",
            "Na2O",
            "K2O",
        ]
    ]
    .dropna()
    .to_numpy()
)
ol_compo = (
    data[
        [
            "P",
            "T",
            "F",
            "SiO2.1",
            "Cr2O3.1",
            "Al2O3.1",
            "FeO.1",
            "MgO.1",
            "CaO.1",
            "MnO.1",
        ]
    ]
    .dropna()
    .to_numpy()
)
opx_compo = (
    data[
        [
            "P",
            "T",
            "F",
            "SiO2.2",
            "TiO2.1",
            "Cr2O3.2",
            "Al2O3.2",
            "FeO.2",
            "MgO.2",
            "CaO.2",
            "MnO.2",
            "Na2O.1",
        ]
    ]
    .dropna()
    .to_numpy()
)
cpx_compo = (
    data[
        [
            "P",
            "T",
            "F",
            "SiO2.3",
            "TiO2.2",
            "Cr2O3.3",
            "Al2O3.3",
            "FeO.3",
            "MgO.3",
            "CaO.3",
            "MnO.3",
            "Na2O.2",
        ]
    ]
    .dropna()
    .to_numpy()
)
gnt_compo = (
    data[
        [
            "P",
            "T",
            "F",
            "SiO2.4",
            "TiO2.3",
            "Cr2O3.4",
            "Al2O3.4",
            "FeO.4",
            "MgO.4",
            "CaO.4",
            "MnO.4",
            "Na2O.3",
        ]
    ]
    .dropna()
    .to_numpy()
)

# Order SiO2 TiO2 Cr2O3 Al2O3 FeO MgO CaO MnO Na2O K2O
molar_mass = np.array(
    [60.08, 79.866, 151.9904, 101.96, 71.844, 40.304, 56.0774, 70.9374, 61.979, 94.196]
)
oxy_per_oxide = np.array([2, 2, 3, 3, 1, 1, 1, 1, 1, 1])
cat_per_oxide = np.array([1, 1, 2, 2, 1, 1, 1, 1, 2, 2])

melt_mask = np.ones_like(molar_mass, dtype=bool)
ol_mask = np.ones_like(molar_mass, dtype=bool)
ol_mask[[1, -2, -1]] = 0
px_mask = np.ones_like(molar_mass, dtype=bool)
px_mask[-1] = 0
gnt_mask = px_mask

melt_mole_numb = melt_compo[:, 3:] / molar_mass
melt_molar_frac = (
    melt_mole_numb / np.tile(melt_mole_numb.sum(axis=1), (melt_mole_numb.shape[1], 1)).T
)

melt_compo_six_oxy = compo_per_oxygen(
    6, melt_compo, melt_mask, molar_mass, oxy_per_oxide, cat_per_oxide
)
ol_compo_four_oxy = compo_per_oxygen(
    4, ol_compo, ol_mask, molar_mass, oxy_per_oxide, cat_per_oxide
)
opx_compo_six_oxy = compo_per_oxygen(
    6, opx_compo, px_mask, molar_mass, oxy_per_oxide, cat_per_oxide
)
cpx_compo_six_oxy = compo_per_oxygen(
    6, cpx_compo, px_mask, molar_mass, oxy_per_oxide, cat_per_oxide
)
gnt_compo_twelve_oxy = compo_per_oxygen(
    12, gnt_compo, gnt_mask, molar_mass, oxy_per_oxide, cat_per_oxide
)

melt_spl = melt_compo[:, 0] < 2
ol_spl = ol_compo[:, 0] < 2
opx_spl = opx_compo[:, 0] < 2
cpx_spl = cpx_compo[:, 0] < 2
gnt_spl = gnt_compo[:, 0] < 2

melt_mgo_numb = melt_compo[:, 8] / (melt_compo[:, 7] + melt_compo[:, 8])
melt_ti = melt_compo_six_oxy[:, 1]
melt_mg = melt_compo_six_oxy[:, 5]
melt_mg_numb = melt_compo_six_oxy[:, 5] / (
    melt_compo_six_oxy[:, 4] + melt_compo_six_oxy[:, 5]
)

ol_mg_numb = ol_compo_four_oxy[:, 4] / (
    ol_compo_four_oxy[:, 3] + ol_compo_four_oxy[:, 4]
)

opx_mg_numb = opx_compo_six_oxy[:, 5] / (
    opx_compo_six_oxy[:, 4] + opx_compo_six_oxy[:, 5]
)
opx_si_t = opx_compo_six_oxy[:, 0]
opx_al_t = np.minimum(2 - opx_si_t, opx_compo_six_oxy[:, 3])
opx_al_m1 = opx_compo_six_oxy[:, 3] - opx_al_t
opx_ti_m1 = opx_compo_six_oxy[:, 1]
opx_cr_m1 = opx_compo_six_oxy[:, 2]
opx_mn_m1 = opx_compo_six_oxy[:, 7] / 2
opx_mn_m2 = opx_compo_six_oxy[:, 7] / 2
opx_ca_m2 = opx_compo_six_oxy[:, 6]
opx_na_m2 = opx_compo_six_oxy[:, 8]
opx_mg_m2 = np.minimum(
    (1 - opx_mn_m2 - opx_ca_m2 - opx_na_m2) * opx_mg_numb, opx_compo_six_oxy[:, 5]
)
opx_fe_m2 = np.minimum(
    (1 - opx_mn_m2 - opx_ca_m2 - opx_na_m2) * (1 - opx_mg_numb), opx_compo_six_oxy[:, 4]
)
opx_mg_m1 = opx_compo_six_oxy[:, 5] - opx_mg_m2
opx_fe_m1 = opx_compo_six_oxy[:, 4] - opx_fe_m2

assert_allclose(
    opx_mn_m1 + opx_al_m1 + opx_ti_m1 + opx_cr_m1 + opx_mg_m1 + opx_fe_m1,
    1,
    rtol=0,
    atol=0.025,
)

cpx_mg_numb = cpx_compo_six_oxy[:, 5] / (
    cpx_compo_six_oxy[:, 4] + cpx_compo_six_oxy[:, 5]
)
cpx_si_t = cpx_compo_six_oxy[:, 0]
cpx_al_t = np.minimum(2 - cpx_si_t, cpx_compo_six_oxy[:, 3])
cpx_al_m1 = cpx_compo_six_oxy[:, 3] - cpx_al_t
cpx_ti_m1 = cpx_compo_six_oxy[:, 1]
cpx_cr_m1 = cpx_compo_six_oxy[:, 2]
cpx_mn_m1 = cpx_compo_six_oxy[:, 7] / 2
cpx_mn_m2 = cpx_compo_six_oxy[:, 7] / 2
cpx_ca_m2 = cpx_compo_six_oxy[:, 6]
cpx_na_m2 = cpx_compo_six_oxy[:, 8]
cpx_mg_m2 = np.minimum(
    (1 - cpx_mn_m2 - cpx_ca_m2 - cpx_na_m2) * cpx_mg_numb, cpx_compo_six_oxy[:, 5]
)
cpx_fe_m2 = np.minimum(
    (1 - cpx_mn_m2 - cpx_ca_m2 - cpx_na_m2) * (1 - cpx_mg_numb), cpx_compo_six_oxy[:, 4]
)
cpx_mg_m1 = cpx_compo_six_oxy[:, 5] - cpx_mg_m2
cpx_fe_m1 = cpx_compo_six_oxy[:, 4] - cpx_fe_m2

assert_allclose(
    cpx_mn_m1 + cpx_al_m1 + cpx_ti_m1 + cpx_cr_m1 + cpx_mg_m1 + cpx_fe_m1,
    1,
    rtol=0,
    atol=0.015,
)

fig, ax = plt.subplots(
    nrows=3,
    ncols=3,
    # figsize=(18, 8),
    figsize=(12, 10),
    # constrained_layout=True,
    subplot_kw={"projection": "3d"},
)
# fig.subplots_adjust(left=0, right=0.99, bottom=0.02, top=1, wspace=0.02, hspace=0.03)
fig.subplots_adjust(left=0, right=0.99, bottom=0.02, top=1, wspace=0, hspace=0.05)

# plot_panel_3d(ax[0, 0], "Olivine Al", ol_spl, ol_compo, ol_compo_four_oxy[:, 2])
# plot_panel_3d(ax[0, 1], "Olivine Mg#", ol_spl, ol_compo, ol_mg_numb)
# plot_panel_3d(ax[0, 2], "Opx Al T", opx_spl, opx_compo, opx_al_t)
# plot_panel_3d(ax[0, 3], "Opx Ca M2", opx_spl, opx_compo, opx_ca_m2)
# plot_panel_3d(ax[1, 0], "Opx Fe M1", opx_spl, opx_compo, opx_fe_m1)
# plot_panel_3d(ax[1, 1], "Opx Mg M1", opx_spl, opx_compo, opx_mg_m1)
# plot_panel_3d(ax[1, 2], "Opx Mg M2", opx_spl, opx_compo, opx_mg_m2)
# plot_panel_3d(ax[1, 3], "Melt Ti", melt_spl, melt_compo, melt_ti)

plot_panel_3d(ax[0, 0], "Cpx Al M1", cpx_spl, cpx_compo, cpx_al_m1)
plot_panel_3d(ax[0, 1], "Cpx Al T", cpx_spl, cpx_compo, cpx_al_t)
plot_panel_3d(ax[0, 2], "Melt Al2O3", melt_spl, melt_compo, melt_molar_frac[:, 3])
plot_panel_3d(ax[1, 0], "Cpx Ca M2", cpx_spl, cpx_compo, cpx_ca_m2)
plot_panel_3d(ax[1, 1], "Cpx Mg#", cpx_spl, cpx_compo, cpx_mg_numb)
plot_panel_3d(ax[1, 2], "Cpx Mg M2", cpx_spl, cpx_compo, cpx_mg_m2)
plot_panel_3d(ax[2, 0], "Melt MgO", melt_spl, melt_compo, melt_compo[:, 8])
plot_panel_3d(ax[2, 1], "Melt SiO2", melt_spl, melt_compo, melt_molar_frac[:, 0])
plot_panel_3d(ax[2, 2], "Grt Ca", gnt_spl, gnt_compo, gnt_compo_twelve_oxy[:, 6])

fig.savefig("atom_mole_oxygen_cpx_grt.pdf", dpi=300)  # , bbox_inches="tight")
# plt.show()
