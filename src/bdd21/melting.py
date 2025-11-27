from dataclasses import InitVar, dataclass

from numpy import clip, isreal
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

cpx_exhaust_coeff = {}  # Keys -> source depletion, from primitive to depleted mantle
cpx_exhaust_coeff[0.0] = [0.14623529, 0.02031577, 0.00267938]
cpx_exhaust_coeff[0.1] = [0.14017647, 0.02358436, 0.00224305]
cpx_exhaust_coeff[0.2] = [0.14279412, 0.01998125, 0.00267615]
cpx_exhaust_coeff[0.3] = [0.13835294, 0.02050388, 0.00267292]
cpx_exhaust_coeff[0.4] = [0.13632353, 0.02116839, 0.00253717]
cpx_exhaust_coeff[0.5] = [0.13517647, 0.02272721, 0.00224305]
cpx_exhaust_coeff[0.6] = [0.13117647, 0.02238655, 0.00235294]
cpx_exhaust_coeff[0.7] = [0.12867647, 0.02207337, 0.00240789]
cpx_exhaust_coeff[0.8] = [0.12870588, 0.02038138, 0.00259858]
cpx_exhaust_coeff[0.9] = [0.12529412, 0.02279994, 0.00218164]
cpx_exhaust_coeff[1.0] = [0.12117647, 0.02489205, 0.00191338]


@dataclass
class PeridotiteMelting:
    """Hydrous peridotite melting parameterisation.

    Mathematical framework originally described in
    Katz et al. - Geochemistry, Geophysics, Geosystems (2003).
    Corrections and updates to the framework described in
    Duvernay et al. - Geochemistry, Geophysics, Geosystems (2021) and
    Duvernay et al. - Geochemistry, Geophysics, Geosystems (2024).
    """

    src_depletion: InitVar[float | None] = None
    original_params: bool = False

    X_H2O_bulk: float = 0.0

    A1: float = 1374.42637742
    A2: float = 141.38033617
    A3: float = -6.82439927
    B1: float = 1688.67306771
    B2: float = 99.49773072
    B3: float = -4.72456844
    C1: float = 2018.83055
    C2: float = 32.3540649
    C3: float = -0.0467952062
    β1: float = 1.5
    β2: float = 1.2
    K: float = 43.0
    γ: float = 0.75
    D_H2O: float = 0.01
    χ1: float = 12.0
    χ2: float = 1.0
    λ: float = 0.6

    c_P: float = 1000.0
    α_s: float = 4e-5
    α_f: float = 6.8e-5
    ρ_s: float = 3300.0
    ρ_f: float = 2900.0
    ΔS: float = 300.0

    def __post_init__(self, src_depletion):
        if self.original_params and src_depletion is None:
            self.A1, self.A2, self.A3 = 1085.7 + 273.15, 132.9, -5.1
            self.B1, self.B2, self.B3 = 1475.0 + 273.15, 80.0, -3.2
            self.C1, self.C2, self.C3 = 1780.0 + 273.15, 45.0, -2.0
            self.M_cpx = 0.17
            self.r0, self.r1 = 0.5, 0.08
            self.β2 = 1.5
        elif not self.original_params and src_depletion is not None:
            self.D1, self.D2, self.D3 = cpx_exhaust_coeff[src_depletion]
        else:
            raise RuntimeError(
                "Either set original_params to True or provide src_depletion."
            )

    def check_water_saturation(
        self, pressure, temp, F, F_cpx_out, T_cpx_out, T_sol, T_liq_lherz, T_liq
    ):
        """Equation 17."""
        X_H2O_sat = self.χ1 * pressure**self.λ + self.χ2 * pressure
        if self.water_content(F) > X_H2O_sat:  # Saturated melt
            if F < F_cpx_out:
                scaled_temp = (temp - T_sol + self.K * X_H2O_sat**self.γ) / (
                    T_liq_lherz - T_sol
                )
                if scaled_temp <= 0:  # T' must be positive for F to be real
                    return 0
                return scaled_temp**self.β1
            else:
                scaled_temp = (temp - T_cpx_out + self.K * X_H2O_sat**self.γ) / (
                    T_liq - T_cpx_out
                )
                if scaled_temp <= 0:  # T' must be positive for F to be real
                    return 0
                return F_cpx_out + (1 - F_cpx_out) * scaled_temp**self.β2
        return F

    def param_vars(self, pressure):
        """Equations 4, 5, 10, 6 + 7, and 9."""
        T_sol = self.A1 + self.A2 * pressure + self.A3 * pressure**2
        T_liq_lherz = self.B1 + self.B2 * pressure + self.B3 * pressure**2
        T_liq = self.C1 + self.C2 * pressure + self.C3 * pressure**2
        if self.original_params:
            F_cpx_out = self.M_cpx / (self.r0 + self.r1 * pressure)
        else:
            F_cpx_out = clip(self.D1 + self.D2 * pressure + self.D3 * pressure**2, 0, 1)
        T_cpx_out = F_cpx_out ** (1 / self.β1) * (T_liq_lherz - T_sol) + T_sol
        return T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out

    def water_content(self, F):
        """Equation 18."""
        return self.X_H2O_bulk / (self.D_H2O + F * (1 - self.D_H2O))

    def equilibrium_batch_melting(self, pressure, temp):
        """Calculates melt fraction according to equilibrium batch melting.

        Section 2
        """

        def root_bracket(accuracy):
            """Determine bracket of values within which the melt fraction lies"""
            F = 0
            for i in range(accuracy):  # Progressively decrease the increment
                increment = 10 ** -(i + 1)

                root_function = root_cpx if F < F_cpx_out else root_opx
                reference_residual = root_function(F)
                if reference_residual is None:
                    return

                while F < 1:
                    F = round(F + increment, i + 1)
                    root_function = root_cpx if F < F_cpx_out else root_opx
                    residual = root_function(F)
                    if residual is None:
                        break
                    elif isreal(residual):
                        if residual * reference_residual <= 0:  # Change of sign
                            if F - increment < F_cpx_out <= F:
                                cpx_out_residual = root_cpx(F_cpx_out)
                                if residual * cpx_out_residual <= 0:
                                    return [F_cpx_out, F]
                                else:
                                    return [F - increment, F_cpx_out]
                            else:
                                return [F - increment, F]
                        elif F == 1:
                            return F
                    else:  # Break to decrease increment
                        break
                F -= increment

        def root_cpx(F):
            """Equations 2 and 3 of Katz et al. (2003) in the context of root finding"""
            scaled_temp = (temp - T_sol + self.K * self.water_content(F) ** self.γ) / (
                T_liq_lherz - T_sol
            )
            if scaled_temp <= 0:  # T' must be positive for F to be real
                return
            return scaled_temp**self.β1 - F

        def root_opx(F):
            """Equations 8 and 9 of Katz et al. (2003) in the context of root finding"""
            scaled_temp = (
                temp - T_cpx_out + self.K * self.water_content(F) ** self.γ
            ) / (T_liq - T_cpx_out)
            if scaled_temp <= 0:  # T' must be positive for F to be real
                return
            return F_cpx_out + (1 - F_cpx_out) * scaled_temp**self.β2 - F

        if pressure > 8:  # Assume parameterisation does not apply
            return 0
        T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out = self.param_vars(pressure)

        bracket = root_bracket(15)  # Accurate bracketing of the melt fraction root
        if bracket is None:  # Complex value, assume below solidus
            return 0
        elif isinstance(bracket, float):  # F is equal to 1
            return bracket
        else:
            F = root_scalar(
                root_cpx if bracket[1] < F_cpx_out else root_opx,
                method="brentq",
                bracket=bracket,
            ).root
            F = 1 if F > 1 else F
            F = self.check_water_saturation(
                pressure, temp, F, F_cpx_out, T_cpx_out, T_sol, T_liq_lherz, T_liq
            )
            assert 0 <= F <= 1
            return F

    def pressure_release_melting(
        self, pres_start, pres_end, temp_start, melt_frac_start, dTdP_GPa
    ):
        """Calculate melt fraction using Section 6 of Katz et al. (2003); the
        temperature gradient along the melting path is supplied as an input variable"""

        def deriv(t, y):
            pressure = t
            temp, F = y
            T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out = self.param_vars(pressure)
            if F <= 0:  # Update F in case solidus has been crossed
                F = self.equilibrium_batch_melting(pressure, temp)
            else:  # Check water saturation
                F = self.check_water_saturation(
                    pressure, temp, F, F_cpx_out, T_cpx_out, T_sol, T_liq_lherz, T_liq
                )
            if F == 0:  # Below solidus
                return dTdP_GPa, 0
            # Derivatives with respect to pressure
            dT_sol = self.A2 + 2 * self.A3 * pressure
            dT_liq_lherz = self.B2 + 2 * self.B3 * pressure
            dT_liq = self.C2 + 2 * self.C3 * pressure
            if self.original_params:
                dF_cpx_out = -(F_cpx_out**2) * self.r1 / self.M_cpx
            else:
                dF_cpx_out = self.D2 + 2 * self.D3 * pressure
            # Derivative correction accounting for the presence of water
            dH2O = (
                self.γ
                * self.K
                * self.X_H2O_bulk**self.γ
                * (1 - self.D_H2O)
                / (self.D_H2O + F * (1 - self.D_H2O)) ** (self.γ + 1)
            )
            if F < F_cpx_out:  # cpx present
                # Equation 22 (corrected) of Katz et al. (2003)
                dTdP_F = F ** (1 / self.β1) * (dT_liq_lherz - dT_sol) + dT_sol
                # Equation 21 (corrected) of Katz et al. (2003)
                dTdF_P = (
                    F ** ((1 - self.β1) / self.β1) * (T_liq_lherz - T_sol) / self.β1
                    + dH2O
                )
            else:  # cpx exhausted
                # Break the pressure derivative of temperature into multiple terms
                A = ((F - F_cpx_out) / (1 - F_cpx_out)) ** (1 / self.β2)
                B = T_liq - T_cpx_out
                dAdP_F = (
                    -dF_cpx_out
                    / self.β2
                    * (F - F_cpx_out) ** (1 / self.β2)
                    * (1 / (F - F_cpx_out) - 1 / (1 - F_cpx_out))
                    / (1 - F_cpx_out) ** (1 / self.β2)
                )
                dCdP_F = (
                    F_cpx_out ** ((1 - self.β1) / self.β1)
                    * dF_cpx_out
                    / self.β1
                    * (T_liq_lherz - T_sol)
                    + F_cpx_out ** (1 / self.β1) * (dT_liq_lherz - dT_sol)
                    + dT_sol
                )
                dBdP_F = dT_liq - dCdP_F
                # Equivalent of Equation 22 of Katz et al. (2003)
                dTdP_F = dAdP_F * B + A * dBdP_F + dCdP_F
                # Equivalent of Equation 21 of Katz et al. (2003)
                dTdF_P = (F - F_cpx_out) ** ((1 - self.β2) / self.β2) / self.β2 * (
                    T_liq - T_cpx_out
                ) / (1 - F_cpx_out) ** (1 / self.β2) + dH2O
            # Equation 20 (modified) of Katz et al. (2003)
            dFdP_S = (dTdP_GPa - dTdP_F) / (temp * self.ΔS / self.c_P + dTdF_P)
            # Equation 23 (modified) of Katz et al. (2003)
            dTdP_S = dTdP_GPa - self.ΔS * dFdP_S * temp / self.c_P
            return dTdP_S, dFdP_S

        # Integrate the coupled system of ordinary differential equations
        sol = solve_ivp(
            deriv,
            [pres_start, pres_end],
            [temp_start, melt_frac_start],
            method="LSODA",
            dense_output=True,
            rtol=1e-13,
            atol=1e-6,
        )

        return sol.sol
