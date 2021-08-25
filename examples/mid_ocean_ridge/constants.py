#!/usr/bin/env python

# List of constant values for simulations:
domain_dim = (1980e3, 660e3)
box_width  = 1980e3
box_depth  = 660e3
T_surf     = 273.
Tm         = 1598.
T_base     = Tm+250.
#Vx         = 1.27e-09 # 4 cm/yr
#Vx         = 3.171e-10 # 1 cm/yr
#Vx         = 6.35e-10 # 2 cm/yr
Vx         = 6.659e-10 # 2.1 cm/yr
#Vplume     = 1.5855e-09 # 5 cm/yr
#Vplume     = 3.171e-09 # 10 cm/yr
kappa      = 1.0e-6
rho0       = 3300.
g          = 9.81
alpha      = 3.0e-5
delta_x    = 1.0e3
Cp         = 1187.

# Melting:
adGra      = 4e-4
DH2O       = 1e-2
XH2Obulk   = 1e-2
mod_cpx    = 0.18
melt_entropy = 407.
r0         = 0.94
r1         = -0.1
B1         = 1520. + 273.15
beta2      = 1.2

# Rheology:
mu_max = 1.0e25
mu_min = 1.0e18
gas_constant = 8.3145

# Diffusion creep
Ediff_UM=300.0e3
Vdiff_UM=4.0e-6
Adiff_UM=3.0e-11

# Dislocationn creep
n=3.5
Adisl_UM=5.0e-16
Edisl_UM=540.0e3
Vdisl_UM=16.0e-6


