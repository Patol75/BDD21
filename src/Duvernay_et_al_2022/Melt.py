#!/usr/bin/env python3
from functools import wraps
import matplotlib.pyplot as plt
from numpy import clip, empty, exp, isreal, linspace
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from time import perf_counter


# Implementation of the hydrous peridote melting parameterisation described in
# Katz et al. (2003)
class Katz(object):
    def __init__(self):  # Parameters included in Katz et al. (2003)
        self.A1, self.A2, self.A3 = 1374.42637742, 141.38033617, -6.82439927
        self.B1, self.B2, self.B3 = 1688.67306771, 99.49773072, -4.72456844
        self.C1, self.C2, self.C3 = 2018.83055, 32.3540649, -0.0467952062
        # Primitive mantle (eNd = 0)
        self.D1, self.D2, self.D3 = 0.14623529, 0.02031577, 0.00267938
        # Equal mix (eNd = 5)
        # self.D1, self.D2, self.D3 = 0.13517647, 0.02272721, 0.00224305
        # Depleted mantle (eNd = 10)
        # self.D1, self.D2, self.D3 = 0.12117647, 0.02489205, 0.00191338
        self.beta1, self.beta2 = 1.5, 1.2
        self.D_H2O, self.X_H2O_bulk = 0.01, 0
        self.gam, self.K = 0.75, 43
        self.ki1, self.ki2 = 12, 1
        self.lam = 0.6
        self.c_P = 1000  # Only for KatzPTF
        self.alpha_s, self.alpha_f = 4e-5, 6.8e-5  # Only for KatzPTF
        self.rho_s, self.rho_f = 3300, 2900  # Only for KatzPTF
        self.deltaS = 300  # Only for KatzPTF

    def updateConst(KatzFunc):  # Decorator to update parameter values
        @wraps(KatzFunc)
        def KatzFuncWrapper(*args, **kwargs):
            if kwargs.get('inputConst'):
                for key, value in kwargs['inputConst'].items():
                    args[0].__dict__[key] = value
                del kwargs['inputConst']
            return KatzFunc(*args, **kwargs)
        return KatzFuncWrapper

    def calcX_H2O(self, F):  # Equation 18 of Katz et al. (2003)
        return self.X_H2O_bulk / (self.D_H2O + F * (1 - self.D_H2O))

    # Equations 4, 5, 10, 6 + 7 and 9 of Katz et al. (2003)
    def calcSolLiqCpxOut(self, presGPa):
        T_sol = self.A1 + self.A2 * presGPa + self.A3 * presGPa ** 2
        T_liq_lherz = self.B1 + self.B2 * presGPa + self.B3 * presGPa ** 2
        T_liq = self.C1 + self.C2 * presGPa + self.C3 * presGPa ** 2
        F_cpx_out = clip(self.D1 + self.D2 * presGPa
                         + self.D3 * presGPa ** 2, 0, 1)
        T_cpx_out = (F_cpx_out ** (1 / self.beta1) * (T_liq_lherz - T_sol)
                     + T_sol)
        return T_sol, T_liq_lherz, T_liq, F_cpx_out, T_cpx_out

    # Equation 17 of Katz et al. (2003)
    def checkWaterSat(self, presGPa, temp, F, T_sol, T_liq_lherz):
        X_H2O_sat = self.ki1 * presGPa ** self.lam + self.ki2 * presGPa
        if self.calcX_H2O(F) > X_H2O_sat:  # Saturation
            # Equations 2 and 3 of Katz et al. (2003), including the
            # temperature correction described in their Section 2.2
            tempPrime = ((temp - T_sol + self.K * X_H2O_sat ** self.gam)
                         / (T_liq_lherz - T_sol))
            F = tempPrime ** self.beta1 if tempPrime > 0 else 0
        return F

    # Calculate melt fraction using Section 2 of Katz et al. (2003)
    @updateConst
    def KatzPT(self, presGPa, temp):
        # Equation 3 of Katz et al. (2003), including the temperature
        # correction described in their Section 2.2; cpx present
        def checkCpx(F):
            return ((temp - T_sol + self.K * self.calcX_H2O(F) ** self.gam)
                    / (T_liq_lherz - T_sol))

        # Equivalent of T' within Equation 8 of Katz et al. (2003), including
        # the temperature correction described in their Section 2.2; cpx
        # exhausted
        def checkOpx(F):
            return ((temp - T_cpx_out + self.K * self.calcX_H2O(F) ** self.gam)
                    / (T_liq - T_cpx_out))

        # Determine bracket of values within which the melt fraction lies
        def detBracket(start, stop, nbIncrmt, funcCheck, func):
            F = start
            for i in range(nbIncrmt):  # Progressively decrease the increment
                incrmt = 10 ** -(i + 1)  # Increment for F
                funcCheckStart = funcCheck(start)  # Calculate initial T'
                if funcCheckStart <= 0:  # T' must be positive for F to be real
                    return None
                # Initial value for root finding
                funcStart = func(start, funcCheckStart)
                while F < stop:
                    funcCheckF = funcCheck(F)  # Calculate T'
                    if funcCheckF >= 0:
                        # Current value for root finding
                        funcF = func(F, funcCheckF)
                        if isreal(funcF):
                            if funcF * funcStart < 0:  # Change of sign
                                return [F - incrmt, F]
                            else:  # No change of sign, increment F
                                F += incrmt
                        else:  # F is complex, break to decrease increment
                            break
                    else:  # T' must be positive for F to be real
                        break
                F -= incrmt
            return [F, F + incrmt]

        # Equation 2 of Katz et al. (2003) in the context of root finding
        def funcCpx(F, *args):
            checkCpxF = args[0] if args else checkCpx(F)
            return F - checkCpxF ** self.beta1

        # Equation 8 of Katz et al. (2003) in the context of root finding
        def funcOpx(F, *args):
            checkOpxF = args[0] if args else checkOpx(F)
            return F - F_cpx_out - (1 - F_cpx_out) * checkOpxF ** self.beta2

        if presGPa > 8:  # Assume parameterisation does not apply
            return 0
        (T_sol, T_liq_lherz, T_liq,
         F_cpx_out, T_cpx_out) = self.calcSolLiqCpxOut(presGPa)
        # Determine accurate bracket of melt fraction values within which the
        # sought melt fraction lies
        bracket = detBracket(0, F_cpx_out, 9, checkCpx, funcCpx)
        if bracket is None:  # Complex value, return 0
            return 0
        elif bracket[1] < F_cpx_out:  # cpx present
            F = root_scalar(funcCpx, method='brentq', bracket=bracket).root
            if F > 0:  # Check water saturation
                F = self.checkWaterSat(presGPa, temp, F, T_sol, T_liq_lherz)
        else:  # cpx exhausted
            bracket = detBracket(F_cpx_out, 1, 9, checkOpx, funcOpx)
            if bracket[1] >= 1:  # Limit melt fraction to 1
                return 1
            F = root_scalar(funcOpx, method='brentq', bracket=bracket).root
        assert 0 <= F < 1
        return F

    # Calculate melt fraction using Section 6 of Katz et al. (2003), modified
    # such that the temperature gradient along the melting path is supplied as
    # an input variable
    @updateConst
    def KatzPTF(self, presGPaStart, presGPaEnd, tempStart, Fstart, dTdP_GPa):
        def deriv(t, y):
            presGPa = t
            temp, F = y
            (T_sol, T_liq_lherz, T_liq,
             F_cpx_out, T_cpx_out) = self.calcSolLiqCpxOut(presGPa)
            if F <= 0:  # Update F in case solidus has been crossed
                F = self.KatzPT(presGPa, temp)
            elif F < F_cpx_out:  # Check water saturation
                F = self.checkWaterSat(presGPa, temp, F, T_sol, T_liq_lherz)
            if F == 0:  # Below solidus
                return dTdP_GPa, 0
            # Derivatives with respect to pressure
            dT_sol = self.A2 + 2 * self.A3 * presGPa
            dT_liq_lherz = self.B2 + 2 * self.B3 * presGPa
            dT_liq = self.C2 + 2 * self.C3 * presGPa
            dF_cpx_out = self.D2 + 2 * self.D3 * presGPa
            # Derivative correction accounting for the presence of water
            dH2O = (self.gam * self.K * self.X_H2O_bulk ** self.gam
                    * (1 - self.D_H2O)
                    / (self.D_H2O + F * (1 - self.D_H2O)) ** (self.gam + 1))
            if F < F_cpx_out:  # cpx present
                # Equation 22 (corrected) of Katz et al. (2003)
                dTdP_F = (F ** (1 / self.beta1) * (dT_liq_lherz - dT_sol)
                          + dT_sol)
                # Equation 21 (corrected) of Katz et al. (2003)
                dTdF_P = (F ** ((1 - self.beta1) / self.beta1)
                          * (T_liq_lherz - T_sol) / self.beta1 + dH2O)
            else:  # cpx exhausted
                # Break the derivative of temperature with respect to pressure
                # into multiple terms
                A = ((F - F_cpx_out) / (1 - F_cpx_out)) ** (1 / self.beta2)
                B = T_liq - T_cpx_out
                dAdP_F = (-dF_cpx_out / self.beta2
                          * (F - F_cpx_out) ** (1 / self.beta2)
                          * (1 / (F - F_cpx_out) - 1 / (1 - F_cpx_out))
                          / (1 - F_cpx_out) ** (1 / self.beta2))
                dCdP_F = (F_cpx_out ** ((1 - self.beta1) / self.beta1)
                          * dF_cpx_out / self.beta1
                          * (T_liq_lherz - T_sol)
                          + F_cpx_out ** (1 / self.beta1)
                          * (dT_liq_lherz - dT_sol) + dT_sol)
                dBdP_F = dT_liq - dCdP_F
                # Equivalent of Equation 22 of Katz et al. (2003)
                dTdP_F = dAdP_F * B + A * dBdP_F + dCdP_F
                # Equivalent of Equation 21 of Katz et al. (2003)
                dTdF_P = ((F - F_cpx_out) ** ((1 - self.beta2) / self.beta2)
                          / self.beta2 * (T_liq - T_cpx_out)
                          / (1 - F_cpx_out) ** (1 / self.beta2) + dH2O)
            # Equation 20 (modified) of Katz et al. (2003)
            dFdP_S = ((dTdP_GPa - dTdP_F)
                      / (temp * self.deltaS / self.c_P + dTdF_P))
            # Equation 23 (modified) of Katz et al. (2003)
            dTdP_S = dTdP_GPa - self.deltaS * dFdP_S * temp / self.c_P
            return dTdP_S, dFdP_S

        # Integrate the coupled system of ordinary differential equations
        sol = solve_ivp(deriv, [presGPaStart, presGPaEnd], [tempStart, Fstart],
                        method='LSODA', dense_output=True,
                        atol=1e-5, rtol=1e-4)
        return sol.sol


# Use existing methods within the below class to produce figures comparable to
# those presented in Katz et al. (2003) and, thereby, assert that the
# implementation of the parameterisation yields correct results
class KatzFigures(object):
    @staticmethod
    def Figure2():
        temp = linspace(1000, 2000, 1001) + 273.15
        F = empty(temp.shape)
        katz = Katz()
        for pres in [0, 1, 2, 3]:
            begin = perf_counter()
            for i, T in enumerate(temp):
                F[i] = katz.KatzPT(pres, T)
            print(perf_counter() - begin)
            plt.plot(temp - 273.15, F, label=f'{pres} GPa')
        plt.legend(loc='upper left')
        plt.grid()
        plt.gca().set_xlim(temp[0] - 273.15, temp[-1] - 273.15)
        plt.gca().set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Figure3():
        pres = linspace(8, 0, 401)
        temp = empty(pres.shape)
        katz = Katz()
        for water in [0, 0.05, 0.1, 0.3, 0.5]:
            begin = perf_counter()
            for i, P in enumerate(pres):
                T = 900 + 273.15
                while katz.KatzPT(P, T, inputConst={'X_H2O_bulk': water}) == 0:
                    T += 0.5
                temp[i] = T - 0.25
            print(perf_counter() - begin)
            plt.plot(temp - 273.15, pres, label=f'{water} bulk wt%')
        plt.legend(loc='upper right')
        plt.grid()
        plt.gca().set_xlim(1000 - 1.2 / 1.8 * 200, 1800 + 1.1 / 1.8 * 200)
        plt.gca().set_ylim(pres[0], pres[-1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Figure4():
        temp = linspace(900, 1400, 501) + 273.15
        F = empty(temp.shape)
        katz = Katz()
        for water in [0, 0.02, 0.05, 0.1, 0.3]:
            begin = perf_counter()
            for i, T in enumerate(temp):
                F[i] = katz.KatzPT(1, T, inputConst={'X_H2O_bulk': water})
            print(perf_counter() - begin)
            plt.plot(temp - 273.15, F, label=f'{water} bulk wt%')
        plt.legend(loc='upper left')
        plt.grid()
        plt.gca().set_xlim(temp[0] - 273.15, temp[-1] - 273.15)
        plt.gca().set_ylim(0, 0.4)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def Figure11():
        fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, sharey=True)
        presGPa = linspace(0, 6, 1001)
        katz = Katz()
        for poTemp, colour in zip([1250, 1350, 1450],
                                  ['tab:blue', 'tab:green', 'tab:orange']):
            temp = (poTemp + 273.15) * exp(katz.alpha_s * 6e9
                                           / katz.c_P / katz.rho_s)
            for water in linspace(0, 0.02, 4):
                begin = perf_counter()
                sol = katz.KatzPTF(6, 0, temp, 0, katz.alpha_s * temp
                                   / katz.rho_s / katz.c_P * 1e9,
                                   inputConst={'X_H2O_bulk': water,
                                               'M_cpx': 0.1})
                print(perf_counter() - begin)
                ax.plot(sol(presGPa)[1, :], presGPa, color=colour)
                bx.plot(sol(presGPa)[0, :] - 273.15, presGPa, color=colour)
        ax.set_ylim((0, 6))
        ax.set_xlim((0, 0.26))
        bx.set_xlim((1180, 1580))
        ax.invert_yaxis()
        for axis in [ax, bx]:
            axis.grid()
            axis.xaxis.tick_top()
        plt.tight_layout()
        plt.show()
