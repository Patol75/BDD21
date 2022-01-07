#!/usr/bin/env python3
import h5py
from numba import float64
from numba.types import unicode_type
from numba.typed import Dict
import numpy as np
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from tqdm.contrib.concurrent import process_map

from ChemistryData import quad_poly_coeff
from Melt import Katz
from MeltChemistryCompiled import solid_composition
from MeltChemistryFunctions import non_zero_initial_melt, run_integrator
from constants import (alpha, attrib, c_P, cs_0, domain_dim, eNd, melt_inputs,
                       rho_mantle, T_mantle)


# Generate mineralogy and melt composition for a 1-D adiabatic melting path
def adiabatic_1D_profile():
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
    # Allocate arrays to store results
    cs_store = np.zeros((sum(mask), cs_0.size))
    cl_store = np.zeros((sum(mask), cs_0.size))
    Fn_store, pn_store = np.zeros((sum(mask), 6)), np.zeros((sum(mask), 6))
    # Impose initial condition for concentrations in the solid
    cs_store[0] = cs_0
    # Initialise variables required during integration
    mask_ode, ode_dX = np.ones(cs_0.size, dtype=bool), 1e-6
    X_0, P_0, pn, cs = 0, part_arr["pressure"][0], np.zeros(6), cs_0
    # Calculate initial modes for mineral phases
    Fn_store[0] = solid_composition(X_0, P_0, quad_poly_coeff, eNd)
    # Iterate over each melting step
    for i in range(sum(mask) - 1):
        # Restrict dictionary of arrays to relevant values for the step
        for arr_str, arr in zip(["melt_fraction", "pressure", "temperature"],
                                [F, presGPa, T]):
            part_arr[arr_str] = arr[mask][i:i+2]
        # Calculate mineralogy and chemistry
        mask_ode, ode_dX, Fn, pn, Dn, D_bulk, P_bulk, cs, cl = run_integrator(
            part_arr, mask_ode, ode_dX, X_0, P_0, pn, cs)
        # Store results
        cs_store[i + 1], cl_store[i + 1] = cs, cl
        Fn_store[i + 1], pn_store[i + 1] = Fn, pn
        # Update X_0 and P_0
        X_0, P_0 = part_arr["melt_fraction"][-1], part_arr["pressure"][-1]
        # Interrupt integration loop if all elements are exhausted
        if not mask_ode.any():
            break
    # Calculate integrated concentrations in the liquid as melting progresses
    Cl = (cumulative_trapezoid(cl_store, x=F[mask], axis=0, initial=0)
          / np.tile(F[mask], (cs_0.size, 1)).T)
    # Save results
    np.savez("1D_adiab", compositions=[cs_store, cl_store, Cl],
             PTF=[presGPa[mask], T[mask], F[mask]],
             mineralogy=[Fn_store, pn_store])


# Extract particle attributes relative to the melting path and calculate
# corresponding mineralogy and chemistry
def main(part_id, part_proc_id):
    # Initialise lists to hold relevant particle attributes
    depth, melt_fraction, melt_rate, pressure, temperature = [], [], [], [], []
    # Determine timesteps over which calculation is to be performed
    timesteps = [0, len(h5f.keys())] if tstep_range is None else tstep_range
    # Extract and store particle attributes iteratively
    for timestep in range(*timesteps):
        # Identify particle
        index = np.asarray((h5f[f'Step#{timestep}']['id'][()] == part_id)
                           & (h5f[f'Step#{timestep}']['proc_id'][()]
                              == part_proc_id)).nonzero()[0][0]
        # Extract values
        depth.append(domain_dim[1] - h5f[f'Step#{timestep}']['y'][()][index])
        melt_fraction.append(h5f[f'Step#{timestep}'][f'{attrib}1'][()][index])
        melt_rate.append(h5f[f'Step#{timestep}'][f'{attrib}3'][()][index])
        pressure.append(h5f[f'Step#{timestep}'][f'{attrib}4'][()][index])
        temperature.append(h5f[f'Step#{timestep}'][f'{attrib}5'][()][index])
    # If melting did not occur, return
    if max(melt_rate) == 0:
        return
    # Identify entries where melting occurs
    mask = [x > 0 for x in melt_rate]
    # Include timestep before melting started
    if melt_fraction[0] == 0:
        mask[mask.index(True) - 1] = True
    elif DISCARD_NON_ZERO_INITIAL_MELT:
        return
    elif melt_rate[0] == 0:
        mask[0] = True
    # Define the dictionary of arrays that describe the melting path
    part_arr = Dict.empty(key_type=unicode_type, value_type=float64[:])
    for attr_arr in ["melt_fraction", "pressure", "temperature"]:
        part_arr[attr_arr] = np.array(locals()[attr_arr])[mask]
    # If non-zero initial melts are allowed, update the dictionary of arrays
    if part_arr["melt_fraction"][0] > 0:
        part_arr = non_zero_initial_melt(part_arr, T_mantle)
    # Initialise variables required during integration
    mask_ode = np.ones(cs_0.size, dtype=bool)
    ode_dX, X_0, P_0, pn_0 = 1e-6, 0, part_arr["pressure"][0], np.zeros(6)
    # Calculate mineralogy and chemistry (pick one of the two below strategies)
    # Single integration over the whole melting path, yielding final values
    mask_ode, ode_dX, Fn, pn, cs, cl = run_integrator(
        part_arr, mask_ode, ode_dX, X_0, P_0, pn_0, cs_0)
    # Step-by-step integration, with possibility to save values along the path
    # cs, pn = cs_0, pn_0
    # for i in range(sum(mask) - 1):
    #     for attr_arr in ["melt_fraction", "pressure", "temperature"]:
    #         part_arr[attr_arr] = np.array(locals()[attr_arr])[mask][i:i+2]
    #     mask_ode, ode_dX, Fn, pn, cs, cl = run_integrator(
    #         part_arr, mask_ode, ode_dX, X_0, P_0, pn, cs)
    #     # Update X_0 and P_0
    #     X_0, P_0 = part_arr["melt_fraction"][-1], part_arr["pressure"][-1]

    return part_id, part_proc_id, [timesteps, mask, cs, cl]


# Uncomment the following two lines to run the 1-D adiabatic calculation
# adiabatic_1D_profile()
# assert False

# Range of timesteps to process; either a 2-tuple (upper-bound not included) or
# None (all steps processed)
tstep_range = None
# Set to True to discard particles whose melting rate is zero at the final
# time-step considered; useful for ridge simulations
ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END = True
# Set to True to discard particles initialised above the solidus; useful for
# systems where the melting region depth extent does not change
DISCARD_NON_ZERO_INITIAL_MELT = True

model_path = Path("/home/thomas/Documents/PaddyCode/ChemistryOnParticles")

compositions = {}
h5part = "standard_case.particles.melt_fraction_and_chemistry.h5part"
with h5py.File(model_path / h5part, "r") as h5f:
    last_step = tstep_range[1] - 1 if tstep_range else len(h5f.keys()) - 1
    if ONLY_INCLUDE_PARTICLES_THAT_MELT_AT_END:
        mask = h5f[f"Step#{last_step}"][f'{attrib}3'][()] > 0
    else:
        mask = h5f[f"Step#{last_step}"][f'{attrib}2'][()] > 0
    h5_ids = h5f[f"Step#{last_step}"]["id"][()][mask]
    h5_proc_ids = h5f[f"Step#{last_step}"]["proc_id"][()][mask]
    if __name__ == "__main__":
        for output in process_map(main, h5_ids, h5_proc_ids,
                                  max_workers=16, chunksize=1):
            if output is not None:
                compositions[f"{output[1]}_{output[0]}"] = output[2]
np.savez(model_path / "test", compositions=compositions)
