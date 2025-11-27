#!/usr/bin/env python3
from operator import itemgetter
from time import perf_counter_ns

import numpy as np
from scipy.constants import convert_temperature
from scipy.integrate import cumulative_simpson

from bdd21 import ChemistryIntegrator, PeridotiteMelting
from bdd21.chemistry_data import DM_SS_2004, PM_MS_1995

T_pot = convert_temperature(1350.0, "C", "K")  # Potential temperature
α = 3e-5  # Coefficient of thermal expansion
c_P = 1187.0  # Specific heat at constant pressure
ρ = 3300.0  # Density

adiab_grad = α * T_pot / ρ / c_P * 1e9  # K / GPa

# Elements modelled
elements = ["Cs", "Rb", "Ba", "Th", "U", "Nb", "Ta", "La", "Ce", "Pb", "Pr", "Sr", "Nd"]
elements.extend(["Zr", "Hf", "Sm", "Eu", "Gd", "Tb", "Dy", "Y", "Ho", "Er", "Yb", "Lu"])
elements.extend(["Na", "K", "Ti", "P", "Li", "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn"])

# Initial concentration of chemical elements in the source
src_depletion = 0.0  # Adjust contributions from primitive mantle and depleted mantle
ele_getter = itemgetter(*elements)
cs_0 = (1 - src_depletion) * np.fromiter(ele_getter(PM_MS_1995), dtype=float)
cs_0 += src_depletion * np.fromiter(ele_getter(DM_SS_2004), dtype=float)

# Melting parameterisation arguments (modify from defaults)
melting_args = {
    "src_depletion": src_depletion,
    "X_H2O_bulk": 0.02,
    "c_P": c_P,
    "α_s": α,
    "ρ_s": ρ,
    "ΔS": 407.0,
}

# Define pressure array and initialise temperature and melt fractiom
P = np.linspace(4.0, 0.0, 4001)
T_start = T_pot * np.exp(adiab_grad / T_pot * P[0])
F_start = 0.0

# Calculate temperature and melt fraction along the adiabatic path
begin = perf_counter_ns()
melting_path = PeridotiteMelting(**melting_args).pressure_release_melting(
    P[0], P[-1], T_start, F_start, adiab_grad
)
T, F = melting_path(P)
print((perf_counter_ns() - begin) / 1e6)

# Identify where melting occurs and include last sub-solidus step
mask = F > 0.0
mask[mask.nonzero()[0][0] - 1] = True
# Array of pressure, temperature, and melt fraction along the melting path
PTF = np.vstack((P, T, F))[:, mask]

# Instance to evolve trace element concentrations
integrator = ChemistryIntegrator(
    PTF, elements, cs_0, src_depletion, const_pc=["Cs", "Rb", "Li"]
)
integrator.advance(cs_0)
integrator = ChemistryIntegrator(
    PTF, elements, cs_0, src_depletion, const_pc=["Cs", "Rb", "Li"]
)
begin = perf_counter_ns()
integrator.advance(cs_0)
print((perf_counter_ns() - begin) / 1e6)
print(integrator.step_outputs["conc_lqd"][-1])

cs = cs_0
for i in range(PTF.shape[1] - 1):
    integrator = ChemistryIntegrator(
        PTF[:, i : i + 2], elements, cs, src_depletion, const_pc=["Cs", "Rb", "Li"]
    )
    integrator.advance(cs_0)
    cs = integrator.step_outputs["conc_sld"][-1]
print(integrator.step_outputs["conc_lqd"][-1])

# Array of pressure, temperature, and melt fraction followed by the integrator
P = integrator.step_outputs["P"]
T = integrator.step_outputs["T"]
F = integrator.step_outputs["F"]
PTF = np.vstack((P, T, F))

# Calculate integrated concentrations in the liquid as melting progresses
conc_lqd = np.vstack(integrator.step_outputs["conc_lqd"])
conc_lqd_cumul = cumulative_simpson(conc_lqd.T, x=F, initial=0.0) / PTF[2]
conc_lqd_cumul[:, 0] = 0.0

# Save results
np.savez(
    "adiabatic_decompression",
    T_pot=T_pot,
    adiab_grad=adiab_grad,
    elements=elements,
    PTF=PTF,
    mod_abund=np.vstack(integrator.step_outputs["mod_abund"]),
    conc_sld=np.vstack(integrator.step_outputs["conc_sld"]),
    conc_lqd=conc_lqd,
    conc_lqd_cumul=conc_lqd_cumul.T,
)
