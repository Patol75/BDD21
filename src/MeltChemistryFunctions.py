#!/usr/bin/env python3
from numpy import (arange, array, diff, empty_like, hstack, interp, linspace,
                   logical_xor, ones, pad, zeros)
from scipy.constants import g
from scipy.integrate import LSODA

from ChemistryData import part_coeff, quad_poly_coeff, radii, valency
from Melt import Katz
from MeltChemistryCompiled import (al_phase_select_array, mineralogy,
                                   solid_composition)
from constants import (cs_0, elements, eNd, gnt_out, melt_inputs, rho_mantle,
                       spl_in, T_mantle)


def run_integrator(part_arr, mask_ode, ode_dX, X_0, P_0, cs_old,
                   X_spl_in, X_gnt_out):
    # Right-hand side of the ode system
    # Equations 3 and 4 of White et al. - JGR: Solid Earth (1992)
    def ode_rhs(X, cs):
        if mnrl_outputs.get(X) is None:  # X-value not yet encountered
            Fn, pn, Dn = mineralogy(X, X_0, P_0, D[mask_step], ri[mask_step],
                                    val[mask_step], *extra_args)
            D_bulk, P_bulk = ones(cs.size), ones(cs.size)
            D_bulk[mask_step], P_bulk[mask_step] = Dn @ Fn, Dn @ pn
            mnrl_outputs[X] = [Fn, pn, Dn, D_bulk, P_bulk]
        D_bulk, P_bulk = mnrl_outputs[X][3], mnrl_outputs[X][4]
        return cs / (1 - X) - cs / (D_bulk - P_bulk * X)

    # Jacobian of the right-hand side of the ode system
    def ode_jac(X, cs):
        D_bulk, P_bulk = mnrl_outputs[X][3], mnrl_outputs[X][4]
        return (1 / (1 - X) - 1 / (D_bulk - P_bulk * X)).reshape((1, -1))

    mnrl_outputs = {}
    mask_step = ones(mask_ode.sum(), dtype=bool)

    Fn_0_spl, Fn_0_gnt = solid_composition(X_0, P_0, quad_poly_coeff, eNd)
    Fn_0 = al_phase_select_array(X_0, X_spl_in, X_gnt_out, Fn_0_spl, Fn_0_gnt)

    D = array([part_coeff[element] for element in elements])[mask_ode]
    ri = array([radii[element] * 1e-10 for element in elements])[mask_ode]
    val = array([valency[element] for element in elements])[mask_ode]
    extra_args = (quad_poly_coeff, eNd, part_arr, X_spl_in, X_gnt_out)

    dX_step = part_arr['melt_fraction'][-1] - part_arr['melt_fraction'][0]
    first_step = min(ode_dX, dX_step)

    solver = LSODA(ode_rhs, part_arr['melt_fraction'][0], cs_old[mask_ode],
                   part_arr['melt_fraction'][-1], first_step=first_step,
                   min_step=1e-14, rtol=1e-5, atol=cs_old[mask_ode] / 1e6,
                   jac=ode_jac, lband=0, uband=0)
    while solver.status == "running":
        solver.step()

        key_X = min(mnrl_outputs, key=lambda key_val: abs(key_val - solver.t))
        Fn = mnrl_outputs[key_X][0]
        # if logical_xor(Fn_0, Fn).any() and solver.status == "running":
        #     if solver.t_old == part_arr['melt_fraction'][0]:
        #         X_0, Fn_0 = solver.t_old, mnrl_outputs[solver.t_old][0]
        #     else:
        #         X_0, Fn_0 = solver.t, Fn
        if logical_xor(Fn_0, Fn).any():
            X_0, Fn_0 = 0.99 * solver.t, Fn
            P_0 = interp(X_0, part_arr["melt_fraction"], part_arr["pressure"])

        mask_step[solver.y / cs_0[mask_ode] < 1e-6] = False

        if solver.step_size < 1e-10:
            first_deriv_inc = solver._lsoda_solver._integrator.rwork[
                20 + mask_step.size:20 + mask_step.size * 2]
            mask_step[abs(first_deriv_inc / solver.y) > 1e-5] = False
    mask_ode[arange(mask_ode.size)[mask_ode][~mask_step]] = False

    X = solver.t
    key_X = min(mnrl_outputs, key=lambda key_val: abs(key_val - X))
    D_bulk, P_bulk = mnrl_outputs[key_X][3], mnrl_outputs[key_X][4]

    cs, cl = zeros(len(elements)), zeros(len(elements))
    cs[mask_ode] = solver.y[mask_step]
    cl[mask_ode] = cs[mask_ode] * (1 - X) / (D_bulk - P_bulk * X)[mask_step]

    mask_neg = (cs <= 0) | (cl < 0)
    mask_ode[mask_neg], cs[mask_neg], cl[mask_neg] = False, 0, 0

    return mask_ode, solver.step_size, X_0, P_0, mnrl_outputs[key_X][0], cs, cl


def calc_X_spl_in_gnt_out(X_spl_in, X_gnt_out, old_depth, current_depth,
                          part_arr, dTdP_GPa):
    if current_depth <= spl_in <= old_depth:
        # Interpolate melt fraction at which spinel becomes a stable phase
        X_spl_in = interp(spl_in, part_arr["pressure"][::-1] * 1e9 / rho_mantle
                          / g, part_arr["melt_fraction"][::-1])
    elif current_depth <= spl_in and X_spl_in == 1:
        X_spl_in = 0
    # Interpolate melt fraction at which garnet becomes a stable phase; extend
    # melting path if last melt is recorded in the spinel-garnet transition
    if gnt_out < current_depth < spl_in:
        P_gnt_out = rho_mantle * g * gnt_out / 1e9
        # Fictitious, subsequent melting path
        sol = Katz().KatzPTF(
            part_arr["pressure"][-1], P_gnt_out, part_arr["temperature"][-1],
            part_arr["melt_fraction"][-1], dTdP_GPa, inputConst=melt_inputs)
        T, X_gnt_out = sol(P_gnt_out)
    elif current_depth <= gnt_out <= old_depth:
        X_gnt_out = interp(gnt_out, part_arr["pressure"][::-1] * 1e9
                           / rho_mantle / g, part_arr["melt_fraction"][::-1])
    elif current_depth <= gnt_out and X_gnt_out == 1:
        X_gnt_out = 0
    return X_spl_in, X_gnt_out


def non_zero_initial_melt(part_arr, pot_temp):
    P = linspace(6, part_arr["pressure"][0],
                 int((6 - part_arr["pressure"][0]) / 0.02 + 1))
    # Linearly distribute cooling of potential temperature over last GPa; 51
    # pressures for an increment of 0.02 GPa in 1 GPa
    T = part_arr["temperature"][0] + pad(linspace(T_mantle - pot_temp, 0, 51),
                                         (P.size - 51, 0),
                                         constant_values=T_mantle - pot_temp)
    X = empty_like(P)
    for i in range(P.size):  # Fictitious, prior melting path
        X[i] = Katz().KatzPT(P[i], T[i], inputConst=melt_inputs)
    assert X[0] == 0 and X[-1] == part_arr["melt_fraction"][0]

    mask = [x > 0 for x in X]  # Identify when melting occurs
    mask[mask.index(True) - 1] = True  # Include entry before solidus
    mask[-1] = False  # Remove duplicate
    part_arr["pressure"] = hstack((P[mask], part_arr["pressure"]))
    part_arr["temperature"] = hstack((T[mask], part_arr["temperature"]))
    part_arr["melt_fraction"] = hstack((X[mask], part_arr["melt_fraction"]))
    assert all(diff(part_arr["melt_fraction"]) > 0)

    return part_arr
