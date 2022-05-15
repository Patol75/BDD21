#!/usr/bin/env python3
from MeltChemistryCompiled import mineralogy, solid_composition
from numpy import asarray, diff, empty_like, linspace, ones, pad, zeros
from scipy.integrate import LSODA

from ChemistryData import mnrl_mode_coeff, part_coeff, radii, valence
from constants import cs_0, elements, melt_inputs, ɛNd
from Melt import Katz


# Calculate mineralogy and chemistry given a melting path
def run_integrator(part_arr, ode_dX, cs_old):
    # Right-hand side of the ode system
    # Equations 3 and 4 of White et al. - JGR: Solid Earth (1992)
    def ode_rhs(X, cs):
        if mnrl_outputs.get(X) is None:  # X-value not yet encountered
            Fn, pn, Dn = mineralogy(
                X,
                X_0,
                Fn_0,
                mask_ode[mask_step],
                D[mask_step],
                ri[mask_step],
                val[mask_step],
                *extra_args,
            )
            D_bulk, P_bulk = ones(cs.size), ones(cs.size)
            D_bulk[mask_step], P_bulk[mask_step] = Dn @ Fn, Dn @ pn

            # Dn_store = zeros((cs.size, 6))
            # Dn_store[mask_step] = Dn

            mnrl_outputs[X] = [Fn, pn, Dn, D_bulk, P_bulk]

        D_bulk, P_bulk = mnrl_outputs[X][3], mnrl_outputs[X][4]
        return cs / (1 - X) - cs / (D_bulk * (1 - X_0) - P_bulk * (X - X_0))

    # Jacobian of the right-hand side of the ode system
    def ode_jac(X, cs):
        D_bulk, P_bulk = mnrl_outputs[X][3], mnrl_outputs[X][4]
        return (1 / (1 - X) - 1 / (D_bulk * (1 - X_0) - P_bulk * (X - X_0))).reshape(
            (1, -1)
        )

    mnrl_outputs = {}
    mask_ode = cs_old.nonzero()[0]
    mask_step = ones(mask_ode.size, dtype=bool)

    X_0, P_0 = part_arr["melt_fraction"][0], part_arr["pressure"][0]
    Fn_0, melting_step = solid_composition(X_0, P_0, mnrl_mode_coeff, ɛNd), 1

    D = asarray([part_coeff[element] for element in elements])[mask_ode]
    ri = asarray([radii[element] * 1e-10 for element in elements])[mask_ode]
    val = asarray([valence[element] for element in elements])[mask_ode]
    extra_args = (mnrl_mode_coeff, ɛNd, part_arr)

    dX_step = part_arr["melt_fraction"][-1] - part_arr["melt_fraction"][0]
    first_step = min(ode_dX, dX_step)

    solver = LSODA(
        ode_rhs,
        part_arr["melt_fraction"][0],
        cs_old[mask_ode],
        part_arr["melt_fraction"][-1],
        first_step=first_step,
        min_step=1e-14,
        rtol=1e-4,
        atol=cs_old[mask_ode] / 1e5,
        jac=ode_jac,
        lband=0,
        uband=0,
    )
    while solver.status == "running":
        solver.step()
        if (
            solver.t > part_arr["melt_fraction"][melting_step]
            and part_arr["melt_fraction"].size > melting_step + 1
        ):
            X_0 = part_arr["melt_fraction"][melting_step]
            P_0 = part_arr["pressure"][melting_step]
            Fn_0 = solid_composition(X_0, P_0, mnrl_mode_coeff, ɛNd)
            melting_step += 1
        # Consider exhausted elements whose normalised concentrations are low
        mask_step[solver.y / cs_0[mask_ode] < 1e-6] = False
        # If the integrator time step collapses, consider that an element is
        # about to exhaust and identify it
        if solver.step_size < 1e-10:
            first_deriv_inc = solver._lsoda_solver._integrator.rwork[
                20 + mask_step.size:20 + mask_step.size * 2
            ]
            mask_step[abs(first_deriv_inc / solver.y) > 1e-4] = False
    mask_ode = mask_ode[mask_step]

    X = solver.t
    key_X = min(mnrl_outputs, key=lambda key_val: abs(key_val - X))
    D_bulk, P_bulk = mnrl_outputs[key_X][3:]

    cs, cl = zeros(len(elements)), zeros(len(elements))
    cs[mask_ode] = solver.y[mask_step]
    cl[mask_ode] = (
        cs[mask_ode] * (1 - X) / (D_bulk * (1 - X_0) - P_bulk * (X - X_0))[mask_step]
    )

    # Dn = zeros((len(elements), 6))
    # D_bulk, P_bulk = zeros(len(elements)), zeros(len(elements))
    # Fn, pn = mnrl_outputs[key_X][:2]
    # Dn[mask_ode] = mnrl_outputs[key_X][2][mask_step]
    # D_bulk[mask_ode] = mnrl_outputs[key_X][3][mask_step]
    # P_bulk[mask_ode] = mnrl_outputs[key_X][4][mask_step]

    mask_neg = (cs <= 0) | (cl < 0)
    cs[mask_neg], cl[mask_neg] = 0, 0

    return solver.step_size, mnrl_outputs[key_X][0], cs, cl


# Update dictionary of arrays based on a fictitious, prior melting path
def non_zero_initial_melt(
    part_arr, part_pot_temp, mant_pot_temp, lab_pressure, dTdP_GPa
):
    step = 0.02
    P = linspace(
        6, part_arr["pressure"][0], int((6 - part_arr["pressure"][0]) / step + 1)
    )

    if part_arr["pressure"][0] > lab_pressure - step:
        T = mant_pot_temp + P * dTdP_GPa
    else:
        distrib_temp_over = int((lab_pressure - part_arr["pressure"][0]) / step + 1)
        T = (
            part_pot_temp
            + P * dTdP_GPa
            + pad(
                linspace(mant_pot_temp - part_pot_temp, 0, distrib_temp_over),
                (P.size - distrib_temp_over, 0),
                constant_values=mant_pot_temp - part_pot_temp,
            )
        )
    T[-1] = part_arr["temperature"][0]

    X = empty_like(P)
    katz = Katz(**melt_inputs)
    for i in range(P.size):  # Fictitious, prior melting path
        X[i] = katz.KatzPT(P[i], T[i])
    assert X[0] == 0 and X[-1] == part_arr["melt_fraction"][0]

    mask = zeros(X.size, dtype=bool)
    max_X = 0
    for i in range(mask.size):
        if X[i] > max_X:
            max_X = X[i]
            mask[i] = True
    mask[mask.nonzero()[0][0] - 1] = True  # Include entry before solidus
    part_arr["pressure"], part_arr["temperature"] = P[mask], T[mask]
    part_arr["melt_fraction"] = X[mask]
    assert all(diff(part_arr["melt_fraction"]) > 0)

    return part_arr
