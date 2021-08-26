#!/usr/bin/env python3

stepWidth = 2e5
domainDim = (4e6, 1e6)
conDepth = 2e5
oceAge = 7e6 * 365.25 * 8.64e4

surfTemp = 2.9e2
mantTemp = 1325 + 273.15

alpha = 3e-5
kappa = 6e-7
R = 8.3145
g = 9.8
rhoMantle = 3.37e3
rhoCraton = 3.3e3
rhoCrust = 2.9e3
adGra = 4e-4

k = 3
d = 9e3
rhoH0 = 6e-6
crusHeatProd = 2.6e-13
mantHeatProd = 4e-15

stepLoc = (1.25e6, 2.75e6)

a0 = 1e-13
r0 = 1.94
e0 = 3.5e5
v0 = 6.8e-6

a1 = 1e-13
r1 = 1.94
e1 = 3.5e5
v1 = 6.8e-6

a2 = 1.2e-16
r2 = 1.94
e2 = 3.5e5
v2 = 2.6e-6

X_H2O_bulk = 2e-2

plateVel = 4e-2 / 8.64e4 / 365.25
poisDepth = 2e5
poisSpeedInc = 0.5

lowMtl = 6.6e5

muMax = 1e24
muMin = 1e18
