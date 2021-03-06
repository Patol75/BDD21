#!/usr/bin/env python3
import numpy as np
from MeltChemistryCompiled import solid_composition
from numba import float64
from numba.typed import Dict
from numba.types import unicode_type
from scipy.integrate import cumulative_trapezoid

from ChemistryData import mnrl_mode_coeff
from constants import T_mantle, alpha, c_P, cs_0, melt_inputs, rho_mantle, ɛNd
from Melt import Katz
from MeltChemistryFunctions import run_integrator

# Initialise pressure and temperature
presGPa = np.linspace(6, 0, 601)
temp = T_mantle * np.exp(alpha / c_P / rho_mantle * presGPa[0] * 1e9)

# Calculate temperature and melt fraction along the adiabatic path
sol = Katz(**melt_inputs).KatzPTF(
    presGPa[0],
    presGPa[-1],
    temp,
    0,
    alpha / c_P / rho_mantle * T_mantle * 1e9,
)
T, F = sol(presGPa)

# Identify entries where melting occurs
mask = [x > 0 for x in F]
mask[mask.index(True) - 1] = True
# Initialise the dictionary of arrays that describe the melting path
part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
for arr_str, arr in zip(["melt_fraction", "pressure", "temperature"], [F, presGPa, T]):
    part_arr[arr_str] = arr[mask]

# Allocate arrays to store results
cs_store = np.zeros((sum(mask), cs_0.size))
cl_store = np.zeros((sum(mask), cs_0.size))
Fn_store = np.zeros((sum(mask), 6))

# Impose initial condition for concentrations in the solid
cs_store[0] = cs_0
# Calculate initial modes for mineral phases
Fn_store[0] = solid_composition(0, part_arr["pressure"][0], mnrl_mode_coeff, ɛNd)

# Initialise variables required during integration
ode_dX, cs = 1e-6, cs_0

# Iterate over each melting step
for i in range(sum(mask) - 1):
    # Restrict dictionary of arrays to relevant values for the step
    for arr_str, arr in zip(
        ["melt_fraction", "pressure", "temperature"], [F, presGPa, T]
    ):
        part_arr[arr_str] = arr[mask][i:i + 2]
    # Calculate mineralogy and chemistry
    ode_dX, Fn, cs, cl = run_integrator(part_arr, ode_dX, cs)
    # Store results
    Fn_store[i + 1], cs_store[i + 1], cl_store[i + 1] = Fn, cs, cl
    # Interrupt integration loop if all elements are exhausted
    if np.count_nonzero(cs) == 0:
        break

# Calculate integrated concentrations in the liquid as melting progresses
Cl = (
    cumulative_trapezoid(cl_store, x=F[mask], axis=0, initial=0)
    / np.tile(F[mask], (cs_0.size, 1)).T
)

# Save results
np.savez(
    "RESULTS",
    compositions=[cs_store, cl_store, Cl],
    PTF=[presGPa[mask], T[mask], F[mask]],
    mineralogy=[Fn_store],
)
