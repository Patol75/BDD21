#!/usr/bin/env python3
from time import perf_counter

import matplotlib.pyplot as plt
from numpy import empty_like, exp, linspace

from bdd21 import PeridotiteMelting


def figure_2(original_params, src_depletion=None):
    temperatures = linspace(1000, 2000, 1001) + 273.15
    F = empty_like(temperatures)
    melting_param = PeridotiteMelting(src_depletion, original_params=original_params)
    for pressure in [0, 1, 2, 3]:
        begin = perf_counter()
        for i, temperature in enumerate(temperatures):
            F[i] = melting_param.equilibrium_batch_melting(pressure, temperature)
        print(perf_counter() - begin)
        plt.plot(temperatures - 273.15, F, label=f"{pressure} GPa")
    plt.legend(loc="upper left")
    plt.grid()
    plt.gca().set_xlim(temperatures[0] - 273.15, temperatures[-1] - 273.15)
    plt.gca().set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def figure_3(original_params, src_depletion=None):
    pressures = linspace(8, 0, 401)
    temperatures = empty_like(pressures)
    melting_param = PeridotiteMelting(src_depletion, original_params=original_params)
    for water in [0, 0.05, 0.1, 0.3, 0.5]:
        begin = perf_counter()
        melting_param.X_H2O_bulk = water
        for i, pressure in enumerate(pressures):
            temperature = 900 + 273.15
            while melting_param.equilibrium_batch_melting(pressure, temperature) == 0:
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


def figure_4(original_params, src_depletion=None):
    temperatures = linspace(900, 1400, 501) + 273.15
    F = empty_like(temperatures)
    melting_param = PeridotiteMelting(src_depletion, original_params=original_params)
    for water in [0, 0.02, 0.05, 0.1, 0.3]:
        begin = perf_counter()
        melting_param.X_H2O_bulk = water
        for i, temperature in enumerate(temperatures):
            F[i] = melting_param.equilibrium_batch_melting(1, temperature)
        print(perf_counter() - begin)
        plt.plot(temperatures - 273.15, F, label=f"{water} bulk wt%")
    plt.legend(loc="upper left")
    plt.grid()
    plt.gca().set_xlim(temperatures[0] - 273.15, temperatures[-1] - 273.15)
    plt.gca().set_ylim(0, 0.4)
    plt.tight_layout()
    plt.show()


def figure_11(original_params, src_depletion=None):
    fig, (ax, bx) = plt.subplots(nrows=1, ncols=2, sharey=True)
    pressures = linspace(0, 6, 1001)
    melting_param = PeridotiteMelting(src_depletion, original_params=original_params)
    melting_param.M_cpx = 0.1
    for potential_temp, colour in zip(
        [1250, 1350, 1450], ["tab:blue", "tab:green", "tab:orange"]
    ):
        temp = (potential_temp + 273.15) * exp(
            melting_param.α_s * 6e9 / melting_param.c_P / melting_param.ρ_s
        )
        dTdP_GPa = (
            melting_param.α_s * temp / melting_param.ρ_s / melting_param.c_P * 1e9
        )
        for water in linspace(0, 0.02, 4):
            begin = perf_counter()
            melting_param.X_H2O_bulk = water
            sol = melting_param.pressure_release_melting(6, 0, temp, 0, dTdP_GPa)
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
