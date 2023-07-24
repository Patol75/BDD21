#!/usr/bin/env python3
from dataclasses import InitVar, dataclass
from time import perf_counter

import matplotlib.pyplot as plt
from numpy import clip, empty_like, exp, isreal, linspace
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


@dataclass
class Katz:
    """Implementation of the hydrous peridotite melting parameterisation described in
    Katz et al. (2003)"""

    src_depletion: InitVar[float] = None
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
        if self.original_params:
            self.A1, self.A2, self.A3 = 1085.7 + 273.15, 132.9, -5.1
            self.B1, self.B2, self.B3 = 1475.0 + 273.15, 80.0, -3.2
            self.C1, self.C2, self.C3 = 1780.0 + 273.15, 45.0, -2.0
            self.M_cpx = 0.17
            self.r0, self.r1 = 0.5, 0.08
            self.β2 = 1.5
        elif src_depletion == 0:  # Primitive mantle
            self.D1, self.D2, self.D3 = 0.14623529, 0.02031577, 0.00267938
        elif src_depletion == 0.1:
            self.D1, self.D2, self.D3 = 0.14017647, 0.02358436, 0.00224305
        elif src_depletion == 0.2:
            self.D1, self.D2, self.D3 = 0.14279412, 0.01998125, 0.00267615
        elif src_depletion == 0.3:
            self.D1, self.D2, self.D3 = 0.13835294, 0.02050388, 0.00267292
        elif src_depletion == 0.4:
            self.D1, self.D2, self.D3 = 0.13632353, 0.02116839, 0.00253717
        elif src_depletion == 0.5:  # Equal mix
            self.D1, self.D2, self.D3 = 0.13517647, 0.02272721, 0.00224305
        elif src_depletion == 0.6:
            self.D1, self.D2, self.D3 = 0.13117647, 0.02238655, 0.00235294
        elif src_depletion == 0.7:
            self.D1, self.D2, self.D3 = 0.12867647, 0.02207337, 0.00240789
        elif src_depletion == 0.8:
            self.D1, self.D2, self.D3 = 0.12870588, 0.02038138, 0.00259858
        elif src_depletion == 0.9:
            self.D1, self.D2, self.D3 = 0.12529412, 0.02279994, 0.00218164
        elif src_depletion == 1:  # Depleted mantle
            self.D1, self.D2, self.D3 = 0.12117647, 0.02489205, 0.00191338
        else:
            raise RuntimeError(
                "Either original_params must be set to True or src_depletion must be"
                " provided as a float in [0, 1]."
            )

    def check_water_saturation(
        self, pressure, temp, F, F_cpx_out, T_cpx_out, T_sol, T_liq_lherz, T_liq
    ):
        """Equation 17 of Katz et al. (2003)"""
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
        """Equations 4, 5, 10, 6 + 7 and 9 of Katz et al. (2003)"""
        T_sol = self.A1 + self.A2 * pressure + self.A3 * pressure**2
        T_liq_lherz = self.B1 + self.B2 * pressure + self.B3 * pressure**2
        T_liq = self.C1 + self.C2 * pressure + self.C3 * pressure**2
        if self.original_params:
            F_cpx_out = self.M_cpx / (self.r0 + self.r1 * pressure)
        else:
            F_cpx_out = clip(
                self.D1 + self.D2 * pressure + self.D3 * pressure**2, 0, 1
            )
        T_cpx_out = F_cpx_out ** (1 / self.β1) * (T_liq_lherz - T_sol) + T_sol
        return T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out

    def water_content(self, F):
        """Equation 18 of Katz et al. (2003)"""
        return self.X_H2O_bulk / (self.D_H2O + F * (1 - self.D_H2O))

    def KatzPT(self, pressure, temp):
        """Calculate melt fraction using Section 2 of Katz et al. (2003)"""

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

        if pressure > 7:  # Assume parameterisation does not apply
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

    def KatzPTF(self, pres_start, pres_end, temp_start, melt_frac_start, dTdP_GPa):
        """Calculate melt fraction using Section 6 of Katz et al. (2003); the
        temperature gradient along the melting path is supplied as an input variable"""

        def deriv(t, y):
            pressure = t
            temp, F = y
            T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out = self.param_vars(pressure)
            if F <= 0:  # Update F in case solidus has been crossed
                F = self.KatzPT(pressure, temp)
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
            atol=1e-5,
            rtol=1e-4,
        )
        return sol.sol


def figure_2(original_params, src_depletion=0):
    temperatures = linspace(1000, 2000, 1001) + 273.15
    F = empty_like(temperatures)
    katz = Katz(src_depletion, original_params=original_params)
    for pressure in [0, 1, 2, 3]:
        begin = perf_counter()
        for i, temperature in enumerate(temperatures):
            F[i] = katz.KatzPT(pressure, temperature)
        print(perf_counter() - begin)
        plt.plot(temperatures - 273.15, F, label=f"{pressure} GPa")
    plt.legend(loc="upper left")
    plt.grid()
    plt.gca().set_xlim(temperatures[0] - 273.15, temperatures[-1] - 273.15)
    plt.gca().set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def figure_3(original_params, src_depletion=0):
    pressures = linspace(8, 0, 401)
    temperatures = empty_like(pressures)
    katz = Katz(src_depletion, original_params=original_params)
    for water in [0, 0.05, 0.1, 0.3, 0.5]:
        begin = perf_counter()
        katz.X_H2O_bulk = water
        for i, pressure in enumerate(pressures):
            temperature = 900 + 273.15
            while katz.KatzPT(pressure, temperature) == 0:
                temperature += 0.5
            temperatures[i] = temperature - 0.25
        print(perf_counter() - begin)
        plt.plot(temperatures - 273.15, pressures, label=f"{water} bulk wt%")
    plt.legend(loc="upper right")
    plt.grid()
    plt.gca().set_xlim(1000 - 1.2 / 1.8 * 200, 1800 + 1.1 / 1.8 * 200)
    plt.gca().set_ylim(pressures[0], pressures[-1])
    plt.tight_layout()
    plt.show()


def figure_4(original_params, src_depletion=0):
    temperatures = linspace(900, 1400, 501) + 273.15
    F = empty_like(temperatures)
    katz = Katz(src_depletion, original_params=original_params)
    for water in [0, 0.02, 0.05, 0.1, 0.3]:
        begin = perf_counter()
        katz.X_H2O_bulk = water
        for i, temperature in enumerate(temperatures):
            F[i] = katz.KatzPT(1, temperature)
        print(perf_counter() - begin)
        plt.plot(temperatures - 273.15, F, label=f"{water} bulk wt%")
    plt.legend(loc="upper left")
    plt.grid()
    plt.gca().set_xlim(temperatures[0] - 273.15, temperatures[-1] - 273.15)
    plt.gca().set_ylim(0, 0.4)
    plt.tight_layout()
    plt.show()


def figure_11(original_params, src_depletion=0):
    fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, sharey=True)
    pressures = linspace(0, 6, 1001)
    katz = Katz(src_depletion, original_params=original_params)
    katz.M_cpx = 0.1
    for potential_temp, colour in zip(
        [1250, 1350, 1450], ["tab:blue", "tab:green", "tab:orange"]
    ):
        temp = (potential_temp + 273.15) * exp(katz.α_s * 6e9 / katz.c_P / katz.ρ_s)
        dTdP_GPa = katz.α_s * temp / katz.ρ_s / katz.c_P * 1e9
        for water in linspace(0, 0.02, 4):
            begin = perf_counter()
            katz.X_H2O_bulk = water
            sol = katz.KatzPTF(6, 0, temp, 0, dTdP_GPa)
            print(perf_counter() - begin)
            ax.plot(sol(pressures)[1, :], pressures, color=colour)
            bx.plot(sol(pressures)[0, :] - 273.15, pressures, color=colour)
    ax.set_ylim((0, 6))
    ax.set_xlim((0, 0.26))
    bx.set_xlim((1180, 1580))
    ax.invert_yaxis()
    for axis in [ax, bx]:
        axis.grid()
        axis.xaxis.tick_top()
    plt.tight_layout()
    plt.show()
