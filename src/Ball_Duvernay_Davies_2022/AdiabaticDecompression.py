#!/usr/bin/env python3
from numba import float64
from numba.types import unicode_type
from numba.typed import Dict
import numpy as np
from scipy.constants import g
from scipy.integrate import cumulative_trapezoid

from ChemistryData import mnrl_mode_coeff
from Melt import Katz
from MeltChemistryCompiled import al_phase_select_array, solid_composition
from MeltChemistryFunctions import calc_X_spl_in_gnt_out, run_integrator
from constants import (adiab_grad, alpha, c_P, cs_0, eNd, melt_inputs,
                       rho_mantle, T_mantle)

# Initialise pressure and temperature
presGPa = np.linspace(6, 0, 601)
temp = T_mantle * np.exp(alpha / c_P / rho_mantle * presGPa[0] * 1e9)

# Calculate temperature and melt fraction along the adiabatic path
sol = Katz().KatzPTF(presGPa[0], presGPa[-1], temp, 0,
                     alpha / c_P / rho_mantle * T_mantle * 1e9,
                     inputConst=melt_inputs)
T, F = sol(presGPa)

# Identify entries where melting occurs
mask = [x > 0 for x in F]
mask[mask.index(True) - 1] = True
# Initialise the dictionary of arrays that describe the melting path
part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
for arr_str, arr in zip(["melt_fraction", "pressure", "temperature"],
                        [F, presGPa, T]):
    part_arr[arr_str] = arr[mask]

# Determine where spinel crystallises and garnet exhausts along the path
depth = part_arr["pressure"] / rho_mantle / g * 1e9
X_spl_in, X_gnt_out = calc_X_spl_in_gnt_out(
    1, 1, depth[0], depth[-1], part_arr,
    adiab_grad / rho_mantle / g * 1e9, rho_mantle)

# Allocate arrays to store results
cs_store = np.zeros((sum(mask), cs_0.size))
cl_store = np.zeros((sum(mask), cs_0.size))
Fn_store = np.zeros((sum(mask), 6))

# Impose initial condition for concentrations in the solid
cs_store[0] = cs_0
# Calculate initial modes for mineral phases
Fn_0_spl, Fn_0_gnt = solid_composition(
    0, part_arr["pressure"][0], mnrl_mode_coeff, eNd)
Fn_store[0] = al_phase_select_array(
    0, X_spl_in, X_gnt_out, Fn_0_spl, Fn_0_gnt)

# Initialise variables required during integration
ode_dX, cs = 1e-6, cs_0

# Iterate over each melting step
for i in range(sum(mask) - 1):
    # Restrict dictionary of arrays to relevant values for the step
    for arr_str, arr in zip(["melt_fraction", "pressure", "temperature"],
                            [F, presGPa, T]):
        part_arr[arr_str] = arr[mask][i:i+2]
    # Calculate mineralogy and chemistry
    ode_dX, Fn, cs, cl = run_integrator(
        part_arr, ode_dX, X_spl_in, X_gnt_out, cs)
    # Store results
    Fn_store[i + 1], cs_store[i + 1], cl_store[i + 1] = Fn, cs, cl
    # Interrupt integration loop if all elements are exhausted
    if np.count_nonzero(cs) == 0:
        break

# Calculate integrated concentrations in the liquid as melting progresses
Cl = (cumulative_trapezoid(cl_store, x=F[mask], axis=0, initial=0)
      / np.tile(F[mask], (cs_0.size, 1)).T)

# Save results
np.savez("RESULTS", compositions=[cs_store, cl_store, Cl],
         PTF=[presGPa[mask], T[mask], F[mask]],
         mineralogy=[Fn_store])
