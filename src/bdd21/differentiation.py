import warnings
from operator import attrgetter

import numpy as np
from bayesbay import State
from scipy.optimize import root


class MagmaticDifferentiation:
    """Predicts the trace element composition of a basalt following differentiation.

    Starting from the composition of a parental magma, the trace-element budget of the
    basalt produced through magmatic differentation is calculated using the equations
    describing a steady-state system, which are derived in Appendix A of [1].

    Attributes
    ----------
    prop_troctolite : float
        The proportion of plutonic rocks in the oceanic's crust layer 3 that formed
        during the first crystallisation event in the chamber (crystallisation of the
        replenishing liquid).

    Parameters
    ----------
    chamber_cycle : str
        The magmatic differentiation cycle to be used. Possible values are:
        "rmtx_albarede", "rmtx", "rmxt", "rtmx", "rxmtx", "rxtmx", "rxmtx_mci",
        "rxtmx_mci", "raxmtx_mci".
    part_coeff : dict[str, list[float]]
        The partition coefficients for the minerals in the chamber. The keys are the
        elements, and the values are lists of partition coefficients for the minerals
        olivine, clinopyroxene, and plagioclase, respectively.
    part_coeff_mg : list[float]
        The partition coefficients for the minerals in the chamber for MgO.
    parental_magma : dict[str, float], optional
        The composition of the parental magma. The keys are the elements, and the
        values are the concentrations of the elements in the parental magma.
    validate : bool, optional
        Whether to validate the crystallisation and tapped fractions. Default is True.
    verbose : bool, optional
        Whether to print the chamber parameters. Default is False.
        This is useful for debugging and understanding the internal state of the
        calculation.

    References
    ----------
    .. [1] Duvernay, T., Jiang, S., Ball, P. W., & Davies, D. R. (2024).
        Coupled geodynamical-geochemical perspectives on the generation and composition
        of mid-ocean ridge basalts.
        Geochemistry, Geophysics, Geosystems, 25(2), e2023GC011288.

    Examples
    --------
    >>> magma_diff = MagmaticDifferentiation(
    ...     chamber_cycle, part_coeff_chamber, part_coeff_chamber_mg, parental_magma
    ... )

    >>> erupted_product = magma_diff.run(chamber_parameters)

    """

    def __init__(
        self,
        chamber_cycle: str,
        part_coeff: dict[str, list[float]],
        part_coeff_mg: list[float],
        parental_magma: dict[str, float] = {},
        *,
        validate: bool = True,
        verbose: bool = False,
    ) -> None:
        self.chamber_cycle = chamber_cycle
        self.part_coeff = part_coeff
        self.part_coeff_mg = part_coeff_mg
        self.parental_magma = parental_magma
        self.validate = validate
        self.verbose = verbose

        self.mnrls = ["ol", "cpx", "pl"]
        self.mci_events = {
            "troctolite": ["asml_1", "melt_1", "tapp_1"],
            "gabbro": ["asml_2", "melt_2", "tapp_2"],
        }

        for param in ["prop_troctolite", "mode_ol_troctolite", "mode_cpx_troctolite"]:
            setattr(self, param, 0.0)
        for mnrl in self.mnrls:
            for events in self.mci_events.values():
                for event in events:
                    setattr(self, f"prop_{mnrl}_{event}", 0.0)

    def chamber_magmatic_fractions(self, ini_guess: tuple[float]) -> tuple[float]:
        """Provides residuals to determine the crystallised and tapped fractions.

        Residuals are calculated using the approach described in [1]. Considering the
        trends of trace element concentrations relative to MgO, there should exist a
        minimum limiting slope for a perfectly incompatible element. This observation
        establishes an equation that can be combined to that of the chamber's cycle; the
        resulting system is applied to MgO to constrain the chamber's crystallised and
        tapped fractions. As the system is non-linear, system residuals are provided to
        a root-finding method (here, the Levenberg-Marquardt algorithm implemented in
        SciPy, [2]) to determine an optimal solution.

        Parameters
        ----------
        ini_guess : tuple of floats
            Initial values of the crystallised and tapped fractions.

        Returns
        -------
        residuals : tuple of floats
            System residuals.

        References
        ----------
        .. [1] St C. O'Neill, H., & Jenner, F. E. (2012).
            The global pattern of trace-element distributions in ocean floor basalts.
            Nature, 491(7426), 698-704.
        .. [2] https://docs.scipy.org/doc/scipy/reference/optimize.root-lm.html

        """
        frac_cryst, frac_tapp = ini_guess

        cycle_rsdl = self.mgo_tapp / self.mgo_repl - self.chamber_differentiation(
            frac_cryst, frac_tapp, self.pc_bulk["mg"], self.β["mg"]
        )  # Cycle equation applied to MgO
        constraint_rsdl = np.log10(1 + frac_cryst / frac_tapp) - self.min_lim_slope * (
            self.mgo_tapp - self.mgo_repl
        )  # Equation A54

        return cycle_rsdl, constraint_rsdl

    def chamber_differentiation(
        self,
        frac_cryst: float,
        frac_tapp: float,
        pc_bulk: list[np.ndarray],
        β: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Calculates the concentration ratio between tapped and replenishing liquids.

        Equations are derived in Appendix A of [1].

        Parameters
        ----------
        frac_tapp : float
            Fraction of tapped liquid
        frac_cryst : float
            Fraction of crystals
        pc_bulk : list of np.ndarray
            Bulk partition coefficient for the crystals
        β : dict of np.ndarray
            Concentration weight of each mineral taking part in each MCI event

        Returns
        -------
        np.ndarray
            Concentration ratio between tapped and replenishing liquids

        References
        ----------
        .. [1] Duvernay, T., Jiang, S., Ball, P. W., & Davies, D. R. (2024).
            Coupled geodynamical-geochemical perspectives on the generation and
            composition of mid-ocean ridge basalts.
            Geochemistry, Geophysics, Geosystems, 25(2), e2023GC011288.
        """
        frac_sum = frac_cryst + frac_tapp

        match self.chamber_cycle:
            case "rmtx_albarede":  # Equation A19
                numerator = frac_sum
                denominator = 1 + frac_tapp - (1 - frac_cryst) ** pc_bulk[1]
            case "rmtx":  # Equation A17
                pfc_term = (1 - frac_cryst / (1 - frac_tapp)) ** pc_bulk[1]

                numerator = frac_sum
                denominator = 1 - (1 - frac_tapp) * pfc_term
            case "rmxt":  # Equation A8
                pfc_term = (1 - frac_cryst) ** (pc_bulk[1] - 1)

                numerator = frac_sum * pfc_term
                denominator = 1 - (1 - frac_sum) * pfc_term
            case "rtmx":  # Equation A13
                pfc_term = (1 - frac_cryst / (1 - frac_tapp)) ** (pc_bulk[1] - 1)

                numerator = frac_sum * pfc_term
                denominator = 1 - frac_tapp - (1 - frac_sum - frac_tapp) * pfc_term
            case "rxmtx":  # NEW
                Φ_x_1 = self.prop_troctolite * frac_cryst
                Φ_x_2 = frac_cryst - Φ_x_1

                Φ_l = 1 - Φ_x_1 - frac_tapp

                γ = 1 - (1 - Φ_x_1 / frac_sum) ** pc_bulk[0]
                δ = 1 - (1 - Φ_x_2 / Φ_l) ** pc_bulk[1]

                numerator = frac_sum * (1 - γ) * frac_tapp
                denominator = frac_tapp * (δ * Φ_l + frac_tapp)
            case "rxtmx":  # Equations A42-43 and A49
                Φ_x_1 = self.prop_troctolite * frac_cryst
                Φ_x_2 = frac_cryst - Φ_x_1

                gamma = 1 - (1 - Φ_x_1 / frac_sum) ** pc_bulk[0]
                frac_avail_cryst_2 = 1 - Φ_x_1 - frac_tapp
                pfc_term = (1 - Φ_x_2 / frac_avail_cryst_2) ** (pc_bulk[1] - 1)

                numerator = frac_sum * (1 - gamma) * pfc_term
                denominator = frac_avail_cryst_2 - (1 - frac_sum - frac_tapp) * pfc_term
            case "rxmtx_mci":  # NEW
                Φ_x_1 = self.prop_troctolite * frac_cryst
                Φ_x_1 /= 1 - self.α["asml_1"] - self.α["melt_1"] - self.α["tapp_1"]
                Φ_x_2 = (1 - self.prop_troctolite) * frac_cryst
                Φ_x_2 /= 1 - self.α["asml_2"] - self.α["melt_2"] - self.α["tapp_2"]

                Φ_l = 1 - (1 - self.α["melt_1"] - self.α["tapp_1"]) * Φ_x_1 - frac_tapp
                Φ_p = frac_sum + self.α["asml_1"] * Φ_x_1 + self.α["asml_2"] * Φ_x_2
                Φ_m = frac_tapp - self.α["tapp_1"] * Φ_x_1 - self.α["tapp_2"] * Φ_x_2

                γ = 1 - (1 - Φ_x_1 / Φ_p) ** pc_bulk[0]
                δ = 1 - (1 - Φ_x_2 / Φ_l) ** pc_bulk[1]
                ε = (1 - (1 - β["melt_1"]) * γ) / (1 - β["asml_1"] * γ)
                η = β["tapp_1"] * γ / (1 - β["asml_1"] * γ)

                beta_term_num = β["tapp_2"] * ε + (1 - β["melt_2"]) * η
                beta_term_den = 1 - β["melt_2"] - β["asml_2"] * ε

                numerator = frac_sum * ((ε + η) * Φ_m + beta_term_num * δ * Φ_l)
                denominator = frac_tapp * (beta_term_den * δ * Φ_l + Φ_m)
            case _:
                raise ValueError(f"Chamber cycle {self.chamber_cycle} not implemented.")

        return numerator / denominator

    def validate_parental(self) -> None:
        # if not self.parental_magma["Nd"] <= 1.354 / 11.2 * self.parental_magma["Zr"]:
        #     raise ValueError

        pass

    def validate_input_parameters(self) -> None:
        for lithology in self.mci_events:
            if any(self.modes[lithology] < 0.0):
                raise ValueError
            if any(self.modes[lithology] > 1.0):
                raise ValueError

        for events in self.mci_events.values():
            if any(sum(self.prop_mnrl[event] for event in events) > 1.0):
                raise ValueError

    def validate_modes(self) -> None:
        if self.prop_troctolite:
            if not 0.2 <= self.modes["troctolite_final"][0] <= 0.6:
                raise ValueError
            if not 0.0 <= self.modes["troctolite_final"][1] <= 0.1:
                raise ValueError
            if not 0.4 <= self.modes["troctolite_final"][2] <= 0.7:
                raise ValueError

            if not 0.0 <= self.modes["gabbro_final"][0] <= 0.25:
                raise ValueError
            if not 0.15 <= self.modes["gabbro_final"][1] <= 0.45:
                raise ValueError
            if not 0.4 <= self.modes["gabbro_final"][2] <= 0.7:
                raise ValueError

        if not 0.05 <= self.modes["layer_3"][0] <= 0.2:
            raise ValueError
        if not 0.2 <= self.modes["layer_3"][1] <= 0.4:
            raise ValueError
        if not 0.5 <= self.modes["layer_3"][2] <= 0.7:
            raise ValueError

    def validate_fractions(self) -> None:
        if not np.isfinite([self.frac_tapp, self.frac_cryst]).all():
            raise ValueError
        if not 0.0 <= self.frac_tapp <= 1.0:
            raise ValueError
        if not 0.0 <= self.frac_cryst <= 1.0:
            raise ValueError
        if not 0.0 <= self.frac_tapp + self.frac_cryst <= 1.0:
            raise ValueError

    def validate_fractions_extra(self) -> None:
        if not 2.0 <= self.frac_cryst / self.frac_tapp <= 4.0:
            raise ValueError

    def set_chamber_parameters(self, parameters: dict[str, float] | State) -> None:
        if isinstance(parameters, dict):
            for param_name, param_val in parameters.items():
                setattr(self, param_name, param_val)
        elif isinstance(parameters, State):
            for space_name, space_state in parameters.param_values.items():
                if space_name == "parental_magma":
                    for p_name, p_val in space_state.param_values.items():
                        self.parental_magma[p_name] = p_val.item()
                else:
                    for p_name, p_val in space_state.param_values.items():
                        setattr(self, p_name, p_val.item())
        else:
            raise TypeError

        self.validate_parental()

        part_coeff_arrays = {}
        part_coeff_arrays["trace"] = np.vstack(
            [self.part_coeff[ele] for ele in self.parental_magma]
        )
        part_coeff_arrays["mg"] = np.asarray(self.part_coeff_mg)

        self.modes = {}
        self.pc_bulk = {"trace": [], "mg": []}
        self.prop_mnrl = {}
        self.α = {}
        self.β = {"trace": {}, "mg": {}}
        for lithology, events in self.mci_events.items():
            mode_ol = getattr(self, f"mode_ol_{lithology}")
            mode_cpx = getattr(self, f"mode_cpx_{lithology}")
            mode_pl = 1.0 - mode_ol - mode_cpx
            self.modes[lithology] = np.array([mode_ol, mode_cpx, mode_pl])

            # if lithology == "gabbro":
            #     part_coeff_arrays["trace"][list(self.parental_magma).index("Sr")][2] = 1.5
            for cat in ["trace", "mg"]:
                self.pc_bulk[cat].append(part_coeff_arrays[cat] @ self.modes[lithology])

            for event in events:
                prop_getter = attrgetter(
                    *[f"prop_{mnrl}_{event}" for mnrl in self.mnrls]
                )
                self.prop_mnrl[event] = np.array(prop_getter(self))
                self.α[event] = np.dot(self.prop_mnrl[event], self.modes[lithology])
                for cat in ["trace", "mg"]:
                    self.β[cat][event] = np.dot(
                        part_coeff_arrays[cat],
                        self.prop_mnrl[event] * self.modes[lithology],
                    )
                    self.β[cat][event] /= self.pc_bulk[cat][-1]

            self.modes[f"{lithology}_final"] = (
                1.0 - sum(self.prop_mnrl[event] for event in events)
            ) * self.modes[lithology]
            self.modes[f"{lithology}_final"] /= self.modes[f"{lithology}_final"].sum()

        self.validate_input_parameters()

        self.modes["layer_3"] = (
            self.prop_troctolite * self.modes["troctolite_final"]
            + (1.0 - self.prop_troctolite) * self.modes["gabbro_final"]
        )

        if self.validate:
            self.validate_modes()

    def print_chamber_parameters(self) -> None:
        print(
            f"Proportion of troctolites (Layer 3): {self.prop_troctolite:.2f}\n"
            f"Troctolite crystallising modes: {self.modes['troctolite'].round(4)}\n"
            f"Troctolite observed modes: {self.modes['troctolite_final'].round(4)}\n"
            f"Gabbro crystallising modes: {self.modes['gabbro'].round(4)}\n"
            f"Gabbro observed modes: {self.modes['gabbro_final'].round(4)}\n"
            f"Layer 3 modes: {self.modes['layer_3'].round(4)}\n"
            f"Crystals assimilated (troctolite): {self.prop_mnrl['asml_1'].round(2)}\n"
            f"Crystals assimilated (gabbro): {self.prop_mnrl['asml_2'].round(2)}\n"
            f"Crystals remelted (troctolite): {self.prop_mnrl['melt_1'].round(2)}\n"
            f"Crystals remelted (gabbro): {self.prop_mnrl['melt_2'].round(2)}\n"
            f"Crystals erupted (troctolite): {self.prop_mnrl['tapp_1'].round(2)}\n"
            f"Crystals erupted (gabbro): {self.prop_mnrl['tapp_2'].round(2)}\n"
            f"Replenishing liquid MgO concentration: {self.mgo_repl:.1f}\n"
            f"Erupted liquid MgO concentration: {self.mgo_tapp:.1f}\n"
            f"Minimum limiting slope: {self.min_lim_slope:.2f}\n"
            f"Fraction crystallised: {self.frac_cryst:.5f}\n"
            f"Fraction erupted: {self.frac_tapp:.5f}\n"
        )

    def run(self, chamber_parameters: dict[str, float] | State) -> np.ndarray:
        self.set_chamber_parameters(chamber_parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.frac_cryst, self.frac_tapp = root(
                self.chamber_magmatic_fractions, (0.2, 0.2), method="lm"
            ).x

        self.validate_fractions()
        if self.validate:
            self.validate_fractions_extra()

        parental_magma_array = np.asarray(list(self.parental_magma.values()))
        erupted_product = parental_magma_array * self.chamber_differentiation(
            self.frac_cryst, self.frac_tapp, self.pc_bulk["trace"], self.β["trace"]
        )

        if self.verbose:
            self.print_chamber_parameters()

        if isinstance(chamber_parameters, State):
            chamber_parameters.save_to_extra_storage("frac_cryst", self.frac_cryst)
            chamber_parameters.save_to_extra_storage("frac_tapp", self.frac_tapp)
            for i, mnrl in enumerate(self.mnrls):
                for lithology in self.mci_events:
                    for suffix in ["", "_final"]:
                        chamber_parameters.save_to_extra_storage(
                            f"mode_{mnrl}_{lithology + suffix}",
                            self.modes[lithology + suffix][i],
                        )
                chamber_parameters.save_to_extra_storage(
                    f"mode_{mnrl}_layer_3", self.modes["layer_3"][i]
                )

        return erupted_product
