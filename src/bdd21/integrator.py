#!/usr/bin/env python3
import numpy as np
from numba.typed import List
from scipy.integrate import LSODA

from .chemistry_data import mnrl_mode_coeff, part_coeff, radii
from .partition import PartitionCoefficients, solid_composition


class ChemistryIntegrator:
    """Integrates the chemistry of a system along a melting path.

    This class uses the LSODA ODE solver to compute the concentrations of elements
    in the solid and liquid phases as a function of melt fraction.
    The integration is performed along a specified pressure, temperature, and melt
    fraction (PTF) path. The class also handles the partition coefficients and
    mineral abundances of the solid phases.

    Attributes:
        PTF (np.ndarray): Pressure, temperature, and melt fraction data.
        src_depletion (float): Source depletion factor.
        element_count (int): Number of elements in the system.
        ind_non_zero_sld (np.ndarray): Indices of non-zero solid concentrations.
        mask_exhausted (np.ndarray): Mask for exhausted elements.
        part_coeff (PartitionCoefficients): Partition coefficients for the elements.
        pc_bulk_tape (dict): Bulk partition coefficients for the current melt fraction.
        step_outputs (dict): Outputs for each integration step, including pressure,
            temperature, melt fraction, modal abundances, solid concentrations,
            and liquid concentrations.
        solver (LSODA): LSODA ODE solver instance.

    Methods:
        reference_step_values(F: float) -> tuple[float, np.ndarray]:
            Determines the current reference melt fraction and modal abundance values.
        ode_rhs(F: float, conc_sld: np.ndarray) -> None:
            Calculates the right-hand side of the ODE system.
        ode_jac(F: float, _) -> None:
            Jacobian of the ODE system's right-hand side.
        advance(conc_sld_ini: np.ndarray) -> None:
            Advances the integration using the LSODA solver.

    Notes:
        The class is designed to work with a specific set of elements and their
        partition coefficients. The initial concentrations of the elements in the
        solid phase should be provided as an array. The integration is performed
        until the melt fraction reaches the last value in the PTF data.

    Parameters:
        PTF (np.ndarray): Pressure, temperature, and melt fraction data.
        elements (list[str]): List of elements to consider.
        conc_sld (np.ndarray): Initial concentrations of the elements in the solid.
        src_depletion (float, optional): Source depletion factor. Defaults to 0.0.
        ode_step (float, optional): Initial step size for the ODE solver. Defaults to 1e-6.
        const_pc (list[str], optional): Elements with constant partition coefficients. Defaults to None.

    Raises:
        AssertionError: If the integration does not reach the last melt fraction in the PTF data.

    Example:
        >>> PTF = np.array([[0, 1], [1000, 2000], [0, 1]])
        >>> elements = ["Si", "Mg", "Fe"]
        >>> conc_sld = np.array([0.5, 0.3, 0.2])
        >>> integrator = ChemistryIntegrator(PTF, elements, conc_sld)
        >>> integrator.advance(conc_sld)
    """

    def __init__(
        self,
        PTF: np.ndarray,
        elements: list[str],
        conc_sld: np.ndarray,
        src_depletion: float = 0.0,
        ode_step: float = 1e-6,
        const_pc: list[str] | None = None,
    ) -> None:
        """Integrates the chemistry of a system along a melting path.

        Args:
            PTF (np.ndarray): Pressure, temperature, and melt fraction data.
            elements (list[str]): List of elements to consider.
            conc_sld (np.ndarray): Initial concentrations of the elements in the solid.
            src_depletion (float, optional): Source depletion factor. Defaults to 0.0.
            ode_step (float, optional): Initial step size for the ODE solver. Defaults to 1e-6.
            const_pc (list[str], optional): Elements with constant partition coefficients. Defaults to None.
        """
        self.PTF = PTF
        self.src_depletion = src_depletion

        self.element_count = len(elements)

        self.ind_non_zero_sld = conc_sld.nonzero()[0]
        self.mask_exhausted = np.ones_like(self.ind_non_zero_sld, dtype=bool)

        elements_present = List([elements[ind] for ind in self.ind_non_zero_sld])
        if const_pc is not None:
            const_pc = List(const_pc)
        self.part_coeff = PartitionCoefficients(
            elements_present, part_coeff, radii, const_ele=const_pc
        )

        self.pc_bulk_tape = {}
        self.step_outputs = {
            "P": [],
            "T": [],
            "F": [],
            "mod_abund": [],
            "conc_sld": [],
            "conc_lqd": [],
        }

        self.solver = LSODA(
            self.ode_rhs,
            self.PTF[2, 0],
            conc_sld[self.ind_non_zero_sld],
            self.PTF[2, -1],
            first_step=min(ode_step, PTF[2, -1] - PTF[2, 0]),
            min_step=1e-14,
            rtol=1e-3,
            atol=0.0,
            jac=self.ode_jac,
            lband=0,
            uband=0,
        )

    def reference_step_values(self, F: float) -> tuple[float, np.ndarray]:
        """Determines current reference melt fraction and modal abundance values."""
        ind_ref_val = np.searchsorted(self.PTF[2], F) - 1
        P_0, F_0 = self.PTF[[0, 2], ind_ref_val]
        mod_abund_0 = solid_composition(P_0, F_0, mnrl_mode_coeff, self.src_depletion)

        return F_0, mod_abund_0

    def ode_rhs(self, F, conc_sld) -> None:
        """Calculates ODE system's right-hand side.

        Equations 3 and 4 of White et al. - JGR: Solid Earth (1992)
        """
        if F not in self.pc_bulk_tape:  # F-value not yet encountered
            # Reference values at the start of the current melting step
            F_0, mod_abund_0 = self.reference_step_values(F)
            # Pressure and temperature linearly interpolated from melt fraction
            P = np.interp(F, self.PTF[2], self.PTF[0])
            T = np.interp(F, self.PTF[2], self.PTF[1])

            # Modal mineral abundances in the residual solid
            mod_abund = solid_composition(P, F, mnrl_mode_coeff, self.src_depletion)
            # Reaction coefficients | Proportions of phases entering the melt
            if F > F_0:
                reac_coeff = (mod_abund_0 * (1 - F_0) - mod_abund * (1 - F)) / (F - F_0)
            else:
                reac_coeff = np.zeros(6)

            # Partition coefficients between all elements and mineral phases considered
            self.part_coeff.update_part_coeff(P, T, F, mod_abund)

            pc_bulk_residual = self.part_coeff.D @ mod_abund
            pc_bulk_reaction = self.part_coeff.D @ reac_coeff
            # Value of 1 - F ensures derivative vanishes for exhausted elements
            pc_bulk = pc_bulk_residual * (1 - F_0) - pc_bulk_reaction * (F - F_0)
            pc_bulk[~self.mask_exhausted] = 1.0 - F
            # Store bulk partition coefficients for the current melt fraction
            self.pc_bulk_tape[F] = pc_bulk

        return conc_sld * (1 / (1 - F) - 1 / self.pc_bulk_tape[F])

    def ode_jac(self, F, _) -> None:
        """Jacobian of the ode system's right-hand side."""
        return 1 / (1 - F) - 1 / self.pc_bulk_tape[F]

    def advance(self, conc_sld_ini: np.ndarray) -> None:
        while self.solver.status == "running":
            self.solver.step()

            # Consider exhausted the elements whose concentrations have fallen below
            # 1e-9 times their initial value
            exhausted = self.solver.y / conc_sld_ini[self.ind_non_zero_sld] < 1e-9
            self.mask_exhausted[exhausted] = False
            # If the integrator time step collapses, assume that an element is about to
            # exhaust and identify it
            if self.solver.step_size < 1e-10:
                first_deriv_inc = self.solver._lsoda_solver._integrator.rwork[
                    20 + self.mask_exhausted.size : 20 + self.mask_exhausted.size * 2
                ]
                self.mask_exhausted[abs(first_deriv_inc / self.solver.y) > 1e-4] = False

            F = self.solver.t

            P = np.interp(F, self.PTF[2], self.PTF[0])
            T = np.interp(F, self.PTF[2], self.PTF[1])
            mod_abund = solid_composition(P, F, mnrl_mode_coeff, self.src_depletion)

            ind_remaining = self.ind_non_zero_sld[self.mask_exhausted]
            conc_sld = np.zeros(self.element_count)
            conc_sld[ind_remaining] = self.solver.y[self.mask_exhausted]

            F_tape = min(self.pc_bulk_tape, key=lambda F_tape: abs(F_tape - F))
            pc_bulk = self.pc_bulk_tape[F_tape]
            conc_lqd = np.zeros_like(conc_sld)
            conc_lqd[ind_remaining] = (
                conc_sld[ind_remaining] * (1 - F) / pc_bulk[self.mask_exhausted]
            )

            self.step_outputs["P"].append(P)
            self.step_outputs["T"].append(T)
            self.step_outputs["F"].append(F)
            self.step_outputs["mod_abund"].append(mod_abund)
            self.step_outputs["conc_sld"].append(conc_sld)
            self.step_outputs["conc_lqd"].append(conc_lqd)

        assert F == self.PTF[2, -1]
