#!/usr/bin/env python3
import h5py
from numba import float64, njit
from numba.types import unicode_type
from numba.typed import Dict
import numpy as np
from pathlib import Path
from scipy.constants import Avogadro, g, pi, R
from scipy.integrate import solve_ivp
from tqdm.contrib.concurrent import process_map
from warnings import catch_warnings, simplefilter

from ChemistryData import (DM_SS_2004, PM_MO_1995,
                           part_coeff, poly_coeffs, radii, valency)
from Melt import Katz


# Wrapper around mineralogy and partition_coefficient to obtain D_bar and P_bar
@njit
def D_bar_and_P_bar(X, D, ri, val, poly_coeffs, eNd, part_arr, X_gnt_out,
                    X_spl_in):
    # Interpolate pressure and temperature from melt fraction
    P = np.interp(X, part_arr["melt_fraction"], part_arr["pressure"])
    T = np.interp(X, part_arr["melt_fraction"], part_arr["temperature"])
    Fn, pn = mineralogy(X, P, poly_coeffs, eNd, part_arr, X_gnt_out, X_spl_in)
    Dn = partition_coefficient(P, T, X, X_gnt_out, X_spl_in, D, ri, val)
    return np.sum(Dn * Fn), np.sum(Dn * pn)


# Select appropriate variable values based on the stability of the aluminous
# phase; linear interpolation is performed within the transition
@njit
def spl_gnt_transition(X, X_spl_in, X_gnt_out, spl_val, gnt_val):
    if X < X_spl_in:  # Garnet stability field
        return gnt_val
    elif X >= X_gnt_out:  # Spinel stability field
        return spl_val
    else:  # Transition
        return ((gnt_val * (X_gnt_out - X) + spl_val * (X - X_spl_in))
                / (X_gnt_out - X_spl_in))


# Calculate proportions of mineral phases remaining in the solid (Fn) and
# entering the melt (pn)
@njit
def mineralogy(X, P, poly_coeffs, eNd, part_arr, X_gnt_out, X_spl_in):
    # Calculation of pn requires the rate of change of Fn to be known. As such,
    # a fictitious, prior melting step needs to be considered.
    if X == 0:  # Melting has yet to begin; one step is sufficient
        X_ind, X, P = 0, np.array([X]), np.array([P])
    else:  # Include fictitious, prior melting step using a small decrement
        dX = X / 10
        X_ind, X = 1, np.array([X - dX, X])
        P = np.interp(X, part_arr["melt_fraction"], part_arr["pressure"])
    # Calculate proportions of parameterised mineral phases; melt is included
    min_comp = {"None": np.zeros_like(X)}  # Proportions of mineral phases
    for mineral, coeffs in poly_coeffs.items():
        amin = coeffs[0, 0] * P ** 2 + coeffs[0, 1] * P + coeffs[0, 2]
        bmin = coeffs[1, 0] * P ** 2 + coeffs[1, 1] * P + coeffs[1, 2]
        min_comp[mineral] = np.minimum(1 - X, np.maximum(amin * X + bmin, 0))
    # Calculate proportions of orthopyroxene for each aluminous phase
    for Al_phase in ["Spl", "Gnt"]:
        Ol, Opx = f"Ol_{Al_phase.lower()}", f"Opx_{Al_phase.lower()}"
        # Assume orthopyroxene makes up for the remaining mineral proportion
        min_comp[Opx] = np.maximum(
            1 - X - min_comp[Ol] - min_comp["Cpx"] - min_comp[Al_phase], 0)
        # Correct proportions in excess
        if (min_comp[Ol] + min_comp["Cpx"] + min_comp[Al_phase] > 1 - X).any():
            assert Al_phase == "Gnt" and (X < X_spl_in).all()
            correction = ((min_comp[Ol] + min_comp["Cpx"] + min_comp[Al_phase])
                          / (1 - X - min_comp[Opx]))
            for min_phase in [Ol, "Cpx", Al_phase]:
                min_comp[min_phase] /= correction
    # Adjust mineral proportions by excluding melt (i.e. solid phases only)
    for mineral, proportion in min_comp.items():
        min_comp[mineral] = proportion / (1 - X)
    # Apply correction from Figure 3 of Kimura and Kawabata (2014)
    KK2014 = 0.04 * eNd / 10
    for Al_phase in ["spl", "gnt"]:
        min_comp["Ol_" + Al_phase] += KK2014
        min_comp["Opx_" + Al_phase] -= np.maximum(KK2014 - min_comp["Cpx"], 0)
    min_comp["Cpx"] -= np.minimum(KK2014, min_comp["Cpx"])
    # Calculate proportions of phases entering the melt
    melt_stoich = {"None": 0.}  # Proportions of phases entering the melt
    for mineral, proportion in min_comp.items():
        if X.size == 2:  # Derived from Equation 4 of Shaw (1979)
            melt_stoich[mineral] = (proportion[0] * (1 - X[1])
                                    - proportion[1] * (1 - X[1] - dX)) / dX
        else:  # Melting has yet to begin; melt_stoich is irrelevant
            melt_stoich[mineral] = 0
    # Form Fn and pn based on which aluminous phases are present
    min_order = [["Ol_spl", "Ol_gnt"], ["Opx_spl", "Opx_gnt"], ["Cpx", "Cpx"],
                 ["None", "None"], ["Spl", "None"], ["None", "Gnt"]]
    Fn, pn = np.empty(6), np.empty(6)
    for i, minerals in enumerate(min_order):
        Fn[i], pn[i] = spl_gnt_transition(
            X[X_ind], X_spl_in, X_gnt_out,
            np.array([min_comp[minerals[0]][X_ind], melt_stoich[minerals[0]]]),
            np.array([min_comp[minerals[1]][X_ind], melt_stoich[minerals[1]]]))
    return Fn, pn


# Calculate partition coefficient using the classical equation from Brice
# (1975) assuming an ideal element
@njit
def Brice1975(Do, E, ro, ri, T):
    return Do * np.exp(-4 * pi * E * Avogadro / R / T
                       * (ro / 2 * (ri - ro) ** 2 + (ri - ro) ** 3 / 3))


# Calculate partition coefficient using the value for another element with the
# same valency
@njit
def Brice1975_element(D_ele, E, ro, ri, r_ele, T):
    return D_ele * np.exp(
        -4 * pi * E * Avogadro / R / T
        * (ro / 2 * (r_ele ** 2 - ri ** 2) + (ri ** 3 - r_ele ** 3) / 3))


# Calculate partition coefficients between a given element and all mineral
# phases considered
@njit
def partition_coefficient(P, T, X, X_gnt_out, X_spl_in, D, ri, val):
    Dn = np.copy(D)  # Initialise Dn with constant values
    # # # # # #
    # Olivine #
    # # # # # #
    if val == 3 and not np.isnan(ri[0]):
        F_Al_gnt, F_Al_spl = 5.64e-3, 1.56e-3
        Mg_num_gnt, Mg_num_spl = 0.07 * X + 0.897, 0.059 * X + 0.904
        F_Al, Mg_num = spl_gnt_transition(X, X_spl_in, X_gnt_out,
                                          np.array([F_Al_spl, Mg_num_spl]),
                                          np.array([F_Al_gnt, Mg_num_gnt]))
        # Equations 17-19 from Sun and Liang (2013)
        Do = np.exp(-0.45 - 0.11 * P + 1.54 * F_Al - 1.94 * Mg_num)
        ro = 0.72e-10
        E = 426e9
        Dn[0] = Brice1975(Do, E, ro, ri[0], T)
    # # # # # # # # #
    # Orthopyroxene #
    # # # # # # # # #
    X_Ca_M2 = -0.756 * X ** 2 + 0.273 * X + 0.063
    X_Mg_M2 = 0.692 * X ** 2 - 0.176 * X + 0.834
    X_Al_T = -0.675 * X ** 2 + 0.041 * X + 0.146
    # Equations 11-13 from Yao et al. (2012)
    Do = np.exp(-5.37 + 3.87e4 / R / T + 3.56 * X_Ca_M2 + 3.54 * X_Al_T)
    ro = 0.69 + 0.43 * X_Ca_M2 + 0.23 * X_Mg_M2
    E = (-1.37 + 1.85 * ro - 0.53 * X_Ca_M2) * 1e12
    if val == 2 and not np.isnan(ri[1]):
        # Section 3.11.10.6.3 of Wood and Blundy (2014)
        D_Mg = 1
        r_Mg = 0.89e-10
        ro_v2 = (ro + 0.08) * 1e-10
        E_v2 = 2 / 3 * E
        Dn[1] = Brice1975_element(D_Mg, E_v2, ro_v2, ri[1], r_Mg, T)
    elif val == 3 and not np.isnan(ri[1]):
        Dn[1] = Brice1975(Do, E, ro * 1e-10, ri[1], T)
    # # # # # # # # #
    # Clinopyroxene #
    # # # # # # # # #
    X_Mg_M2_spl, X_Mg_M2_gnt = 0.583 * X + 0.223, 0.422 * X + 0.547
    X_Al_T_spl, X_Al_T_gnt = -0.177 * X + 0.154, -0.013 * X + 0.061
    X_Al_M1_spl, X_Al_M1_gnt = -0.438 * X + 0.137, -0.114 * X + 0.099
    X_Mg_M1_spl, X_Mg_M1_gnt = 0.425 * X + 0.741, 0.191 * X + 0.793
    X_Mg_Mel_spl, X_Mg_Mel_gnt = 0.14 * X + 0.722, 0.207 * X + 0.701
    X_Mg_M1, X_Mg_M2, X_Al_M1, X_Al_T, X_Mg_Mel = spl_gnt_transition(
        X, X_spl_in, X_gnt_out,
        np.array([X_Mg_M1_spl, X_Mg_M2_spl, X_Al_M1_spl, X_Al_T_spl,
                  X_Mg_Mel_spl]),
        np.array([X_Mg_M1_gnt, X_Mg_M2_gnt, X_Al_M1_gnt, X_Al_T_gnt,
                  X_Mg_Mel_gnt]))
    X_H2O_Mel = 0  # Neglect effect of water in melt
    # Equations 8-10 from Sun and Liang (2012)
    Do = np.exp(-7.14 + 7.19e4 / R / T + 4.37 * X_Al_T + 1.98 * X_Mg_M2
                - 0.91 * X_H2O_Mel)
    ro = 1.066 - 0.104 * X_Al_M1 - 0.212 * X_Mg_M2
    E = (2.27 * ro - 2) * 1e12
    if val == 1 and not np.isnan(ri[1]):
        # Section 3.11.10.1.3 of Wood and Blundy (2014)
        D_Na = np.exp((10_367 + 2100 * P - 165 * P ** 2) / T - 10.27
                      + 0.358 * P - 0.0184 * P ** 2)
        r_Na = 1.18e-10
        ro_v1 = (ro + 0.12) * 1e-10
        E_v1 = E / 3
        Dn[2] = Brice1975_element(D_Na, E_v1, ro_v1, ri[1], r_Na, T)
    elif val == 2 and not np.isnan(ri[1]):
        # Section 3.11.10.1.2 of Wood and Blundy (2014)
        D_Ca = 2  # UPDATE | Hill, Blundy and Wood (2011)
        r_Ca = 1.12e-10
        ro_v2 = (ro + 0.06) * 1e-10
        E_v2 = 2 / 3 * E
        Dn[2] = Brice1975_element(D_Ca, E_v2, ro_v2, ri[1], r_Ca, T)
    elif val == 3 and not np.isnan(ri[1]):
        Dn[2] = Brice1975(Do, E, ro * 1e-10, ri[1], T)
    elif val == 4 and not np.isnan(ri[1]):
        # Section 3.11.10.1.4 of Wood and Blundy (2014)
        r_Th = 1.041e-10
        ro_v4 = ro * 1e-10
        E_v4 = 4 / 3 * E
        Y_Mg_M1 = np.exp(902 * (1 - X_Mg_M1) ** 2 / T)
        Y_Th_M2 = 1 / Brice1975(1, E_v4, ro_v4, r_Th, T)
        D_Th = np.exp((214_790 - 175.7 * T + 16_420 * P - 1500 * P ** 2)
                      / R / T) * X_Mg_Mel / X_Mg_M1 / Y_Mg_M1 / Y_Th_M2
        Dn[2] = Brice1975_element(D_Th, E_v4, ro_v4, ri[1], r_Th, T)
    # # # # # # # #
    # Plagioclase #
    # # # # # # # #
    # # # # # #
    # Spinel  #
    # # # # # #
    # # # # # #
    # Garnet  #
    # # # # # #
    F_Ca = -0.247 * X + 0.355
    # Equations 10-12 from Sun and Liang (2013)
    Do = np.exp(-2.05 + (9.17e4 - 91.35 * P * (38 - P)) / R / T
                - 1.02 * F_Ca)
    ro = 0.78 + 0.155 * F_Ca
    E = (-1.62 + 2.29 * ro) * 1e12
    if val == 2 and not np.isnan(ri[1]):
        # Section 3.11.10.3.3 of Wood and Blundy (2014)
        D_Mg = (np.exp((258_210 - 141.5 * T + 5418 * P) / 3 / R / T)
                / np.exp(1.9e4 * F_Ca ** 2 / R / T))
        r_Mg = 0.89e-10
        ro_v2 = (0.053 + ro) * 1e-10
        E_v2 = 2 / 3 * E
        Dn[2] = Brice1975_element(D_Mg, E_v2, ro_v2, ri[1], r_Mg, T)
    elif val == 3 and not np.isnan(ri[1]):
        Dn[5] = Brice1975(Do, E, ro * 1e-10, ri[1], T)
    elif val == 4 and not np.isnan(ri[0]):
        # Figure 7 of Mallmann and O'Neill (2007)
        Do_v4, ro_v4, E_v4 = 4.38, 0.6626e-10, 2753e9
        Dn[5] = Brice1975(Do_v4, E_v4, ro_v4, ri[0], T)

    return Dn


# Calculate concentrations in the melt and residue based on a pressure,
# temperature, melt fraction path
def particle_composition(part_arr):
    # Equation to integrate to obtain the concentration in the residue as a
    # function of melt fraction; derived from Equations 3 and 4 of White et al.
    # (1992)
    def integrateWhite1992(X, cs, *args):
        if D_bar_dict.get(X) is None:  # X-value not yet encountered
            D_bar_dict[X], P_bar_dict[X] = D_bar_and_P_bar(X, *args)
        return cs / (1 - X) - cs / (D_bar_dict[X] - P_bar_dict[X] * X)

    # Jacobian of the ordinary differential equation
    def jacobianWhite1992(X, cs, *args):
        if D_bar_dict.get(X) is None:  # X-value not yet encountered
            D_bar_dict[X], P_bar_dict[X] = D_bar_and_P_bar(X, *args)
        return 1 / (1 - X) - 1 / (D_bar_dict[X] - P_bar_dict[X] * X)

    # Event function provided to solve_ivp
    def stopIntegration(X, cs, *args):
        return cs[0] - cs_0 / 1e5  # Limit cs to cs_0 / 1e5

    stopIntegration.terminal = True  # Stop integration if a zero is found in
    stopIntegration.direction = -1  # the event function (positive to negative)

    assert np.all(np.diff(part_arr["vert_coord"]) > 0)  # numpy.interp

    # Interpolate melt fraction at which spinel becomes a stable phase
    X_spl_in = np.interp(model_depth - spl_in, part_arr["vert_coord"],
                         part_arr["melt_fraction"], left=0, right=1)

    # Interpolate melt fraction at which garnet becomes a stable phase; extend
    # melting path if last melt is recorded in the spinel-garnet transition
    if gnt_out < model_depth - part_arr["vert_coord"].max() < spl_in:
        assert part_arr["vert_coord"].max() == part_arr["vert_coord"][-1]
        P_gnt_out = inputs["rho_s"] * g * gnt_out / 1e9
        sol = Katz().KatzPTF(  # Fictitious, subsequent melting path
            part_arr["pressure"][-1], P_gnt_out, part_arr["temperature"][-1],
            part_arr["melt_fraction"][-1], inputs["alpha_s"] * Tp
            / inputs["rho_s"] / inputs["c_P"] * 1e9, inputConst=inputs)
        T, X_gnt_out = sol(P_gnt_out)
    else:
        X_gnt_out = np.interp(model_depth - gnt_out, part_arr["vert_coord"],
                              part_arr["melt_fraction"], left=0, right=1)

    if part_arr["melt_fraction"][0] > 0:  # Deal with non-zero initial melt
        # Code within this if block is dependent on how the models used to
        # obtain melting paths were initialised
        adGra = 4e-4
        P = np.linspace(6, part_arr["pressure"][0],
                        int((6 - part_arr["pressure"][0]) / 0.02 + 1))
        Tp_particle = (part_arr["temperature"][0]
                       - adGra * (model_depth - part_arr["vert_coord"][0]))
        # Linearly distribute cooling of potential temperature over last GPa
        T = (Tp_particle + adGra / 9.8 / inputs["rho_s"] * P * 1e9
             + np.pad(np.linspace(Tp - Tp_particle, 0, 51), (P.size - 51, 0),
                      constant_values=Tp - Tp_particle))
        X = np.empty_like(P)
        for i in range(P.size):  # Fictitious, prior melting path
            X[i] = Katz().KatzPT(P[i], T[i], inputConst=inputs)
        assert X[0] == 0 and X[-1] > 0
        mask = [x > 0 for x in X]  # Identify when melting occurs
        mask[mask.index(True) - 1] = True  # Include entry before solidus
        mask[-1] = False  # Discard duplicate
        part_arr["pressure"] = np.hstack((P[mask], part_arr["pressure"]))
        part_arr["temperature"] = np.hstack((T[mask], part_arr["temperature"]))
        part_arr["melt_fraction"] = np.hstack(
            (X[mask], part_arr["melt_fraction"]))
    assert np.all(np.diff(part_arr["melt_fraction"]) > 0)  # numpy.interp

    integration_range = [part_arr["melt_fraction"][0],
                         part_arr["melt_fraction"][-1]]
    cs = np.zeros((part_arr["melt_fraction"].size, len(elements)))
    cl = np.zeros((part_arr["melt_fraction"].size, len(elements)))
    for j, element in enumerate(elements):
        D_bar_dict, P_bar_dict = {}, {}
        cs_0 = ((10 - eNd) / 10 * PM_MO_1995[element]
                + eNd / 10 * DM_SS_2004[element])
        extra_args = (part_coeff[element], radii[element] * 1e-10,
                      valency[element], poly_coeffs, eNd, part_arr,
                      X_gnt_out, X_spl_in)
        with catch_warnings():
            simplefilter('ignore', category=UserWarning)
            sol = solve_ivp(  # Determine cs along melting path
                integrateWhite1992, integration_range, [cs_0], method="LSODA",
                t_eval=part_arr["melt_fraction"], events=stopIntegration,
                args=extra_args, first_step=1e-9, max_step=np.inf, rtol=5e-4,
                atol=5e-7, jac=jacobianWhite1992, lband=0, uband=0,
                min_step=1e-11)
        cs[:sol.y.size, j] = sol.y
        for i, X in enumerate(part_arr["melt_fraction"]):  # Calculate cl
            if X == 0:  # No melt yet
                continue
            if D_bar_dict.get(X) is None:  # X-value not yet encountered
                D_bar_dict[X], P_bar_dict[X] = D_bar_and_P_bar(X, *extra_args)
            cl_step = cs[i, j] * (1 - X) / (D_bar_dict[X] - P_bar_dict[X] * X)
            if 0 < cl_step < 1e6:  # Only allow valid cl values
                cl[i, j] = cl_step
            else:  # Discard all remaining values
                break
    return cs, cl


# Extract particle attributes relative to the melting path and initiate
# concentration calculations
def main(part_id, part_proc_id):
    vert_coord, melt_fraction, melt_rate, pres, temp = [], [], [], [], []
    timesteps = tstep_range if bool(tstep_range) else [0, len(h5f.keys())]
    for timestep in range(*timesteps):  # Extract and store particle attributes
        index = np.asarray((h5f[f'Step#{timestep}']['id'][()] == part_id)
                           & (h5f[f'Step#{timestep}']['proc_id'][()]
                              == part_proc_id)).nonzero()[0][0]
        vert_coord.append(h5f[f'Step#{timestep}']['y'][()][index])
        melt_fraction.append(h5f[f'Step#{timestep}']['Katz1'][()][index])
        melt_rate.append(h5f[f'Step#{timestep}']['Katz3'][()][index])
        pres.append(h5f[f'Step#{timestep}']['Katz4'][()][index])
        temp.append(h5f[f'Step#{timestep}']['Katz5'][()][index])
    if max(melt_rate) == 0 or max(melt_fraction) < 1e-3:  # Insufficient melts
        return
    mask = [x > 0 for x in melt_rate]  # Timesteps when melting is happening
    if melt_fraction[0] == 0:  # Include timestep before melting starts
        mask[mask.index(True) - 1] = True
    elif DISCARD_NON_ZERO_INITIAL_MELT:
        return
    else:  # Only needed if melting rate is initially set to 0
        mask[0] = True
    part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
    part_arr["vert_coord"] = np.asarray(vert_coord)[mask]
    part_arr["melt_fraction"] = np.asarray(melt_fraction)[mask]
    part_arr["pressure"] = np.asarray(pres)[mask]
    part_arr["temperature"] = np.asarray(temp)[mask]
    cs, cl = particle_composition(part_arr)
    return part_id, part_proc_id, [timesteps, mask, cs, cl]


# Generate data for an adiabatic melting path
def adiabatic_1D_profile():
    from scipy.integrate import cumulative_trapezoid

    presGPa = np.linspace(6, 0, 601)
    temp = Tp * np.exp(inputs["alpha_s"] * presGPa[0] * 1e9 / inputs["c_P"]
                       / inputs["rho_s"])
    sol = Katz().KatzPTF(presGPa[0], presGPa[-1], temp, 0, inputs["alpha_s"]
                         * Tp / inputs["rho_s"] / inputs["c_P"] * 1e9,
                         inputConst=inputs)
    T, F = sol(presGPa)
    mask = [x > 0 for x in F]
    mask[mask.index(True) - 1] = True
    part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
    part_arr["melt_fraction"] = F[mask]
    part_arr["vert_coord"] = (model_depth - presGPa[mask] * 1e9
                              / inputs["rho_s"] / g)
    part_arr["temperature"] = T[mask]
    part_arr["pressure"] = presGPa[mask]
    cs, cl = particle_composition(part_arr)
    Cl = np.zeros_like(cl)
    Fn, pn = np.empty((sum(mask), 6)), np.empty((sum(mask), 6))
    X_spl_in = np.interp(model_depth - spl_in, part_arr["vert_coord"],
                         part_arr["melt_fraction"], left=0, right=1)
    X_gnt_out = np.interp(model_depth - gnt_out, part_arr["vert_coord"],
                          part_arr["melt_fraction"], left=0, right=1)
    for i, (X, P) in enumerate(zip(part_arr["melt_fraction"],
                                   part_arr["pressure"])):
        Fn[i], pn[i] = mineralogy(X, P, poly_coeffs, eNd, part_arr, X_gnt_out,
                                  X_spl_in)
    for j in range(len(elements)):
        Cl[:, j] = cumulative_trapezoid(cl[:, j], x=part_arr["melt_fraction"],
                                        initial=0) / part_arr["melt_fraction"]
    np.savez("/OUTPUT/PATH",
             compositions=[cs, cl, Cl], PTF=[presGPa[mask], T[mask], F[mask]],
             mineralogy=[Fn, pn])


# Elements to include within the concentration calculations; each element must
# have an associated valency, radius and partition coefficient list within
# ChemistryData.py
elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Na', 'Ti', 'Hf', 'Rb', 'Sr', 'Th', 'U', 'Pb',
            'Nb', 'Zr', 'Y', 'Ta', 'Sc', 'V', 'Cr', 'K', 'P', 'Ba']
# Parameters to provide to the melting functions; only parameters described in
# Katz et al. (2003) are valid; parameters names must match these defined in
# __init__ of Katz within Melt.py
inputs = {"alpha_s": 3e-5, "B1": 1520 + 273.15, "beta2": 1.2, "c_P": 1187,
          "deltaS": 407, "M_cpx": 0.18, "r0": 0.94, "r1": -0.1,
          "rho_s": 3.3e3, "X_H2O_bulk": 0.01}
# Additional parameters used within the script
model_depth, Tp, spl_in, gnt_out, eNd = 6.6e5, 1598, 70e3, 69e3, 10

assert spl_in >= gnt_out
assert 0 <= eNd <= 10

tstep_range = [100, 201]  # Upper-bound not included; None to use all steps

# Set to True to discard particles whose melting rate is zero at the final
# time-step considered; useful for ridge simulations
ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END = True
# Set to True to discard particles initialised above the solidus; useful for
# systems where the melting region depth extent does not change
DISCARD_NON_ZERO_INITIAL_MELT = True

model_path = Path("/PATH/TO/MODEL")

compositions = {}
with h5py.File(model_path / "PARTICLES.h5part", "r") as h5f:
    last_step = tstep_range[1] - 1 if tstep_range else len(h5f.keys()) - 1
    if ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END:
        mask = h5f[f"Step#{last_step}"]["Katz3"][()] > 0
    else:
        mask = h5f[f"Step#{last_step}"]["Katz2"][()] > 0
    h5_ids = h5f[f"Step#{last_step}"]["id"][()][mask]
    h5_proc_ids = h5f[f"Step#{last_step}"]["proc_id"][()][mask]
    if __name__ == "__main__":
        for output in process_map(main, h5_ids, h5_proc_ids,
                                  max_workers=16, chunksize=1):
            if output is not None:
                compositions[f"{output[1]}_{output[0]}"] = output[2]
np.savez(model_path / "OUTPUT_NAME", compositions=compositions)
