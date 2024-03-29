import warnings
from itertools import product
from multiprocessing import Pool
from re import split
from time import perf_counter

# import matplotlib.pyplot as plt
import numpy as np
from ridge_data import (  # prim_mantle_comp,; bulk_part_coeff,
    elements_ext_ree,
    grid_chem_integ_xy,
    grid_rate_integ_xy,
    missing_elements,
    n_morb_comp,
    part_coeff,
    part_coeff_mg,
)
from scipy.optimize import root


def magma_chamber_dynamics(
    ini_guess, model, bulk_part_coeff_mg, mgo_repl, mgo_tapp, min_lim_slope, *args
):
    frac_tapp, frac_cryst = ini_guess
    return (
        min_lim_slope * (mgo_tapp - mgo_repl) - np.log10(1 + frac_cryst / frac_tapp),
        mgo_tapp / mgo_repl
        - magma_chamber_fractionation(
            frac_tapp, frac_cryst, model, bulk_part_coeff_mg, *args
        ),
    )


def magma_chamber_fractionation(frac_tapp, frac_cryst, model, bulk_part_coeff, *args):
    match model:
        case "rmtx_albarede":
            return (frac_cryst + frac_tapp) / (
                1 + frac_tapp - (1 - frac_cryst) ** bulk_part_coeff
            )
        case "rmtx_o_hara":
            return (frac_cryst + frac_tapp) / (
                1
                - (1 - frac_tapp)
                * (1 - frac_cryst / (1 - frac_tapp)) ** bulk_part_coeff
            )
        case "rmxt":
            return (
                (frac_cryst + frac_tapp)
                * (1 - frac_cryst) ** (bulk_part_coeff - 1)
                / (
                    1
                    - (1 - frac_cryst - frac_tapp)
                    * (1 - frac_cryst) ** (bulk_part_coeff - 1)
                )
            )
        case "rtmx":
            return (
                (frac_cryst + frac_tapp)
                * (1 - frac_cryst / (1 - frac_tapp)) ** (bulk_part_coeff - 1)
                / (
                    1
                    - frac_tapp
                    - (1 - frac_cryst - 2 * frac_tapp)
                    * (1 - frac_cryst / (1 - frac_tapp)) ** (bulk_part_coeff - 1)
                )
            )
        case "rxmtx":
            prop_cryst_repl = args[0]

            frac_cryst_1 = prop_cryst_repl * frac_cryst
            frac_cryst_2 = (1 - prop_cryst_repl) * frac_cryst

            return (
                (frac_cryst + frac_tapp)
                * (1 - frac_cryst_1 / (frac_cryst + frac_tapp)) ** bulk_part_coeff[0]
                / (
                    1
                    - frac_cryst_1
                    - (1 - frac_cryst_1 - frac_tapp)
                    * (1 - frac_cryst_2 / (1 - frac_cryst_1 - frac_tapp))
                    ** bulk_part_coeff[1]
                )
            )
        case "rxtmx":
            prop_cryst_repl = args[0]

            frac_cryst_1 = prop_cryst_repl * frac_cryst
            frac_cryst_2 = (1 - prop_cryst_repl) * frac_cryst

            return (
                (frac_cryst + frac_tapp)
                * (1 - frac_cryst_1 / (frac_cryst + frac_tapp)) ** bulk_part_coeff[0]
                * (1 - frac_cryst_2 / (1 - frac_cryst_1 - frac_tapp))
                ** (bulk_part_coeff[1] - 1)
                / (
                    1
                    - frac_cryst_1
                    - frac_tapp
                    - (1 - frac_cryst_1 - frac_cryst_2 - 2 * frac_tapp)
                    * (1 - frac_cryst_2 / (1 - frac_cryst_1 - frac_tapp))
                    ** (bulk_part_coeff[1] - 1)
                )
            )
        case "rxmtx_mci":
            prop_cryst_repl = args[0]
            cryst_tapp_frac_1 = args[1]
            scaled_conc_fact_tapp_1 = args[2]
            cryst_melt_frac_2 = args[3]
            scaled_conc_fact_melt_2 = args[4]

            frac_cryst_1 = prop_cryst_repl / (1 - cryst_tapp_frac_1) * frac_cryst
            frac_cryst_2 = (1 - prop_cryst_repl) / (1 - cryst_melt_frac_2) * frac_cryst

            return (
                (frac_cryst + frac_tapp)
                * (
                    1
                    + (
                        scaled_conc_fact_tapp_1
                        * (1 - scaled_conc_fact_melt_2)
                        * (1 - (1 - cryst_tapp_frac_1) * frac_cryst_1 - frac_tapp)
                        / (frac_tapp - cryst_tapp_frac_1 * frac_cryst_1)
                        * (
                            1
                            - (
                                1
                                - frac_cryst_2
                                / (
                                    1
                                    - (1 - cryst_tapp_frac_1) * frac_cryst_1
                                    - frac_tapp
                                )
                            )
                            ** bulk_part_coeff[1]
                        )
                        - (1 - scaled_conc_fact_tapp_1)
                    )
                    * (
                        1
                        - (1 - frac_cryst_1 / (frac_cryst + frac_tapp))
                        ** bulk_part_coeff[0]
                    )
                )
                / frac_tapp
                / (
                    1
                    + (1 - scaled_conc_fact_melt_2)
                    * (1 - (1 - cryst_tapp_frac_1) * frac_cryst_1 - frac_tapp)
                    / (frac_tapp - cryst_tapp_frac_1 * frac_cryst_1)
                    * (
                        1
                        - (
                            1
                            - frac_cryst_2
                            / (1 - (1 - cryst_tapp_frac_1) * frac_cryst_1 - frac_tapp)
                        )
                        ** bulk_part_coeff[1]
                    )
                )
            )


def main(parameters):
    (
        prop_troctolite,
        mode_ol_troctolite,
        mode_cpx_troctolite,
        mode_ol_gabbro,
        mode_cpx_gabbro,
        prop_ol_tapp_1,
        prop_cpx_tapp_1,
        prop_pl_tapp_1,
        prop_ol_melt_2,
        prop_cpx_melt_2,
        prop_pl_melt_2,
        mgo_repl,
        mgo_tapp,
        min_lim_slope,
    ) = parameters

    if prop_troctolite == 0 and (
        abs(mode_ol_troctolite - 0.25) > 1e-3 or abs(mode_cpx_troctolite - 0.0) > 1e-3
    ):
        return 1

    simu = "d10_ss"
    model = "rxmtx_mci"
    part_coeff_key = "white_2014"
    part_coeff_mg_key = "laubier_2014"
    n_morb_comp_key = "gale_2013"
    # prim_mantle_comp_key = "sun_1989"

    grid_rate_integ_xy_run = grid_rate_integ_xy[simu]
    grid_chem_integ_xy_run = grid_chem_integ_xy[simu]
    part_coeff_run = part_coeff[part_coeff_key]
    part_coeff_run["Ba"][2] = part_coeff["duvernay_2023"]["Ba"][2]
    part_coeff_run["Pb"][2] = part_coeff["duvernay_2023"]["Pb"][2]
    part_coeff_run["Sr"][2] = part_coeff["duvernay_2023"]["Sr"][2]
    part_coeff_run["Eu"][2] = part_coeff["duvernay_2023"]["Eu"][2]
    part_coeff_mg_run = part_coeff_mg[part_coeff_mg_key]
    bulk_part_coeff_run = {}
    # bulk_part_coeff_run = bulk_part_coeff["o_neill_2012"]

    if "gabbro" in simu:
        elements = [
            ele
            for ele in elements_ext_ree
            if ele in part_coeff_run.keys() and ele not in missing_elements["gabbro"]
            # if ele in bulk_part_coeff_run.keys()
            # and ele not in missing_elements["gabbro"]
        ]
    elif "pyroxenite" in simu:
        elements = [
            ele
            for ele in elements_ext_ree
            if ele in part_coeff_run.keys()
            and ele not in missing_elements["pyroxenite"]
            # if ele in bulk_part_coeff_run.keys()
            # and ele not in missing_elements["pyroxenite"]
        ]
    else:
        simu_key = "_".join(split("_", simu)[:2])
        elements = [
            ele
            for ele in elements_ext_ree
            if ele in part_coeff_run.keys()
            and ele not in missing_elements[simu_key[-2:]]
            # if ele in bulk_part_coeff_run.keys()
            # and ele not in missing_elements[simu_key[-2:]]
        ]
    elements.remove("Sr")

    ele_conc_tapp = np.empty(len(elements))
    ele_conc_gale = np.empty(len(elements))
    # ele_conc_pm = np.empty(len(elements))
    for i, element in enumerate(elements):
        ele_conc_gale[i] = n_morb_comp[n_morb_comp_key][element]
        # ele_conc_pm[i] = prim_mantle_comp[prim_mantle_comp_key][element]

    app_frac_ol_1, app_frac_cpx_1, app_frac_pl_1 = {}, {}, {}
    app_frac_ol_2, app_frac_cpx_2, app_frac_pl_2 = {}, {}, {}
    scaled_conc_fact_tapp_1 = {}
    scaled_conc_fact_melt_2 = {}

    troctolite_modes = np.array(
        [
            mode_ol_troctolite,
            mode_cpx_troctolite,
            1 - mode_ol_troctolite - mode_cpx_troctolite,
        ]
    )

    gabbro_modes = np.array(
        [mode_ol_gabbro, mode_cpx_gabbro, 1 - mode_ol_gabbro - mode_cpx_gabbro]
    )

    prop_mnrl_tapp_1 = np.array([prop_ol_tapp_1, prop_cpx_tapp_1, prop_pl_tapp_1])

    prop_mnrl_melt_2 = np.array([prop_ol_melt_2, prop_cpx_melt_2, prop_pl_melt_2])

    bulk_part_coeff_mg = [
        troctolite_modes @ part_coeff_mg_run,
        gabbro_modes @ part_coeff_mg_run,
    ]
    # bulk_part_coeff_mg = gabbro_modes @ part_coeff_mg_run

    app_frac_ol_mg_1 = troctolite_modes @ [
        1,
        part_coeff_mg_run[1] / part_coeff_mg_run[0],
        part_coeff_mg_run[2] / part_coeff_mg_run[0],
    ]
    app_frac_cpx_mg_1 = troctolite_modes @ [
        part_coeff_mg_run[0] / part_coeff_mg_run[1],
        1,
        part_coeff_mg_run[2] / part_coeff_mg_run[1],
    ]
    app_frac_pl_mg_1 = troctolite_modes @ [
        part_coeff_mg_run[0] / part_coeff_mg_run[2],
        part_coeff_mg_run[1] / part_coeff_mg_run[2],
        1,
    ]

    app_frac_ol_mg_2 = gabbro_modes @ [
        1,
        part_coeff_mg_run[1] / part_coeff_mg_run[0],
        part_coeff_mg_run[2] / part_coeff_mg_run[0],
    ]
    app_frac_cpx_mg_2 = gabbro_modes @ [
        part_coeff_mg_run[0] / part_coeff_mg_run[1],
        1,
        part_coeff_mg_run[2] / part_coeff_mg_run[1],
    ]
    app_frac_pl_mg_2 = gabbro_modes @ [
        part_coeff_mg_run[0] / part_coeff_mg_run[2],
        part_coeff_mg_run[1] / part_coeff_mg_run[2],
        1,
    ]

    scaled_conc_fact_tapp_mg_1 = (
        prop_mnrl_tapp_1
        * troctolite_modes
        @ [1 / app_frac_ol_mg_1, 1 / app_frac_cpx_mg_1, 1 / app_frac_pl_mg_1]
    )

    scaled_conc_fact_melt_mg_2 = (
        prop_mnrl_melt_2
        * gabbro_modes
        @ [1 / app_frac_ol_mg_2, 1 / app_frac_cpx_mg_2, 1 / app_frac_pl_mg_2]
    )

    for element, values in part_coeff_run.items():
        bulk_part_coeff_run[element] = [
            troctolite_modes @ values,
            gabbro_modes @ values,
        ]
        # bulk_part_coeff_run[element] = gabbro_modes @ values

        app_frac_ol_1[element] = troctolite_modes @ [
            1,
            values[1] / values[0],
            values[2] / values[0],
        ]
        app_frac_cpx_1[element] = troctolite_modes @ [
            values[0] / values[1],
            1,
            values[2] / values[1],
        ]
        app_frac_pl_1[element] = troctolite_modes @ [
            values[0] / values[2],
            values[1] / values[2],
            1,
        ]

        app_frac_ol_2[element] = gabbro_modes @ [
            1,
            values[1] / values[0],
            values[2] / values[0],
        ]
        app_frac_cpx_2[element] = gabbro_modes @ [
            values[0] / values[1],
            1,
            values[2] / values[1],
        ]
        app_frac_pl_2[element] = gabbro_modes @ [
            values[0] / values[2],
            values[1] / values[2],
            1,
        ]

        scaled_conc_fact_tapp_1[element] = (
            prop_mnrl_tapp_1
            * troctolite_modes
            @ [
                1 / app_frac_ol_1[element],
                1 / app_frac_cpx_1[element],
                1 / app_frac_pl_1[element],
            ]
        )

        scaled_conc_fact_melt_2[element] = (
            prop_mnrl_melt_2
            * gabbro_modes
            @ [
                1 / app_frac_ol_2[element],
                1 / app_frac_cpx_2[element],
                1 / app_frac_pl_2[element],
            ]
        )

    cryst_tapp_frac_1 = prop_mnrl_tapp_1 @ troctolite_modes
    cryst_melt_frac_2 = prop_mnrl_melt_2 @ gabbro_modes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frac_tapp, frac_cryst = root(
            magma_chamber_dynamics,
            (0.2, 0.2),
            args=(
                model,
                bulk_part_coeff_mg,
                mgo_repl,
                mgo_tapp,
                min_lim_slope,
                prop_troctolite,
                cryst_tapp_frac_1,
                scaled_conc_fact_tapp_mg_1,
                cryst_melt_frac_2,
                scaled_conc_fact_melt_mg_2,
            ),
            method="lm",
        ).x

    if not np.isfinite(frac_tapp) or not np.isfinite(frac_cryst):
        return 1

    if frac_tapp + frac_cryst < 0.01 or frac_tapp + frac_cryst > 0.9:
        return 1

    layer_2_3_ratio = frac_cryst / frac_tapp
    if layer_2_3_ratio < 2.3:
        return 1

    upd_troctolite_modes = troctolite_modes * (1 - prop_mnrl_tapp_1)
    upd_troctolite_modes /= upd_troctolite_modes.sum()
    if upd_troctolite_modes[0] < 0.249 or upd_troctolite_modes[1] > 0.101:
        return 1

    upd_gabbro_modes = gabbro_modes * (1 - prop_mnrl_melt_2)
    upd_gabbro_modes /= upd_gabbro_modes.sum()
    if upd_gabbro_modes[0] > 0.151 or upd_gabbro_modes[2] < upd_gabbro_modes[1]:
        return 1

    layer_3_modes = upd_troctolite_modes * prop_troctolite + upd_gabbro_modes * (
        1 - prop_troctolite
    )
    if layer_3_modes[0] < 0.149 or layer_3_modes[0] > 0.201 or layer_3_modes[2] < 0.499:
        return 1

    for i, element in enumerate(elements):
        ele_conc_tapp[i] = (
            magma_chamber_fractionation(
                frac_tapp,
                frac_cryst,
                model,
                bulk_part_coeff_run[element],
                prop_troctolite,
                cryst_tapp_frac_1,
                scaled_conc_fact_tapp_1[element],
                cryst_melt_frac_2,
                scaled_conc_fact_melt_2[element],
            )
            * grid_chem_integ_xy_run[element]
            / grid_rate_integ_xy_run
        )

    tapp_conc_residual = ((ele_conc_tapp / ele_conc_gale - 1) ** 2).sum() / len(
        elements
    )

    # plt.plot(
    #     elements,
    #     ele_conc_tapp / ele_conc_pm,
    #     linestyle="none",
    #     marker="o",
    #     markersize=8,
    # )
    # plt.plot(
    #     elements,
    #     ele_conc_gale / ele_conc_pm,
    #     linestyle="none",
    #     marker="o",
    #     markersize=8,
    # )
    # plt.gca().set_yscale("log")
    # plt.savefig("test.pdf")

    if tapp_conc_residual < 0.01905:
        print(
            f"{prop_troctolite:.2f} "
            f"{troctolite_modes.round(4)} "
            f"{np.around(upd_troctolite_modes, 4)} "
            f"{gabbro_modes.round(4)} "
            f"{np.around(upd_gabbro_modes, 4)} "
            f"{np.around(layer_3_modes, 4)} "
            f"{prop_ol_tapp_1:.2f} "
            f"{prop_cpx_tapp_1:.2f} "
            f"{prop_pl_tapp_1:.2f} "
            f"{prop_ol_melt_2:.2f} "
            f"{prop_cpx_melt_2:.2f} "
            f"{prop_pl_melt_2:.2f} "
            f"{mgo_repl:.1f} "
            f"{mgo_tapp:.1f} "
            f"{min_lim_slope:.2f} "
            f"{frac_tapp:.5f} "
            f"{frac_cryst:.5f} "
            f"{layer_2_3_ratio:.3f} "
            f"{tapp_conc_residual:.5f}"
        )

        return tapp_conc_residual
    else:
        return 1


begin = perf_counter()

np.set_printoptions(precision=4, floatmode="fixed")

min_res = 1

# prop_troctolite = np.linspace(0.1, 0.3, 6)
# mode_ol_troctolite = np.linspace(0.2, 0.4, 6)
# mode_cpx_troctolite = np.linspace(0.0, 0.1, 6)
# mode_ol_gabbro = np.linspace(0.05, 0.2, 6)
# mode_cpx_gabbro = np.linspace(0.2, 0.35, 6)
# prop_ol_tapp_1 = np.linspace(0.1, 0.3, 5)
# prop_cpx_tapp_1 = np.linspace(0.25, 0.45, 5)
# prop_pl_tapp_1 = np.linspace(0.0, 0.2, 5)
# prop_ol_melt_2 = np.linspace(0.2, 0.4, 5)
# prop_cpx_melt_2 = np.linspace(0.3, 0.5, 5)
# prop_pl_melt_2 = np.linspace(0.6, 0.8, 5)
# mgo_repl = np.linspace(10.0, 10.4, 3)
# mgo_tapp = np.linspace(7.6, 7.8, 3)
# min_lim_slope = np.linspace(-0.24, -0.2, 3)

prop_troctolite = np.linspace(0.2, 0.2, 1)
mode_ol_troctolite = np.linspace(0.25, 0.27, 3)
mode_cpx_troctolite = np.linspace(0.0, 0.02, 3)
mode_ol_gabbro = np.linspace(0.03, 0.05, 3)
mode_cpx_gabbro = np.linspace(0.45, 0.47, 3)
prop_ol_tapp_1 = np.linspace(0.02, 0.04, 3)
prop_cpx_tapp_1 = np.linspace(0.0, 0.1, 2)
prop_pl_tapp_1 = np.linspace(0.02, 0.04, 3)
prop_ol_melt_2 = np.linspace(0.0, 0.2, 11)
prop_cpx_melt_2 = np.linspace(0.70, 0.72, 3)
prop_pl_melt_2 = np.linspace(0.72, 0.74, 3)
mgo_repl = np.linspace(10.2, 10.4, 3)
mgo_tapp = np.linspace(7.6, 7.8, 3)
min_lim_slope = np.linspace(-0.2, -0.2, 1)

# prop_troctolite = np.linspace(0.0, 0.1, 11)
# mode_ol_troctolite = np.linspace(0.25, 0.4, 16)
# mode_cpx_troctolite = np.linspace(0.0, 0.1, 11)
# mode_ol_gabbro = np.linspace(0.13, 0.15, 3)
# mode_cpx_gabbro = np.linspace(0.19, 0.22, 4)
# prop_ol_tapp_1 = np.linspace(0.0, 0.3, 1)
# prop_cpx_tapp_1 = np.linspace(0.0, 0.5, 1)
# prop_pl_tapp_1 = np.linspace(0.0, 0.2, 1)
# prop_ol_melt_2 = np.linspace(0.0, 0.4, 1)
# prop_cpx_melt_2 = np.linspace(0.0, 0.5, 1)
# prop_pl_melt_2 = np.linspace(0.0, 0.8, 1)
# mgo_repl = np.linspace(10.0, 10.7, 8)
# mgo_tapp = np.linspace(7.6, 7.8, 3)
# min_lim_slope = np.linspace(-0.21, -0.19, 3)

if __name__ == "__main__":
    with Pool(processes=104) as pool:
        for output in pool.imap_unordered(
            main,
            product(
                prop_troctolite,
                mode_ol_troctolite,
                mode_cpx_troctolite,
                mode_ol_gabbro,
                mode_cpx_gabbro,
                prop_ol_tapp_1,
                prop_cpx_tapp_1,
                prop_pl_tapp_1,
                prop_ol_melt_2,
                prop_cpx_melt_2,
                prop_pl_melt_2,
                mgo_repl,
                mgo_tapp,
                min_lim_slope,
            ),
            chunksize=300,
        ):
            min_res = min(min_res, output)

print(perf_counter() - begin, min_res)

# prop_troctolite = 0.08
# mode_ol_troctolite = 0.25
# mode_cpx_troctolite = 0.00
# mode_ol_gabbro = 0.13
# mode_cpx_gabbro = 0.25
# prop_ol_tapp_1 = 0.50
# prop_cpx_tapp_1 = 0.20
# prop_pl_tapp_1 = 0.25
# prop_ol_melt_2 = 0.10
# prop_cpx_melt_2 = 0.10
# prop_pl_melt_2 = 0.60
# mgo_repl = 10.1
# mgo_tapp = 7.7
# min_lim_slope = -0.20

# main(
#     (
#         prop_troctolite,
#         mode_ol_troctolite,
#         mode_cpx_troctolite,
#         mode_ol_gabbro,
#         mode_cpx_gabbro,
#         prop_ol_tapp_1,
#         prop_cpx_tapp_1,
#         prop_pl_tapp_1,
#         prop_ol_melt_2,
#         prop_cpx_melt_2,
#         prop_pl_melt_2,
#         mgo_repl,
#         mgo_tapp,
#         min_lim_slope,
#     )
# )

# 0.15 [0.35 0.   0.65] [0.3011 0.     0.6989] [0.1  0.25 0.65] [0.1702 0.3457 0.484 ]
# [0.1898 0.2939 0.5163] 0.20 0.40 0.00 0.20 0.35 0.65 10.2 7.7 -0.21 0.16572 0.38938
# 2.350 0.00948

# 0.24 [0.32 0.03 0.65] [0.2764 0.0217 0.7019] [0.09 0.23 0.68] [0.1749 0.3391 0.486 ]
# [0.1993 0.2629 0.5378] 0.20 0.33 0.00 0.13 0.34 0.68 10.2 7.7 -0.21 0.17220 0.40460
# 2.350 0.00919
