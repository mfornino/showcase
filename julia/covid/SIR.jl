# Simple SIR Model for COVID-19 Modeling
# Michele Fornino
# 3/23/2020
#
# Based on Atkeson (2020)

# Housekeeping
using Plots
using DifferentialEquations

# Define System of Differential Equations
function SIRE!(du, u, p, t)
  # Order of variables in vectors u and du:
  # u[1] S(usceptible)
  # u[2] E(xposed)
  # u[3] I(nfected)
  # u[4] R(recovered/dead)
  #
  # Order of parameters in vector p:
  # p[1] γ
  # p[2] σ

  # Define auxiliary variable
  S, E, I, R = u
  γ, σ, R0 = p
  N = sum(u)
  β = R0 * γ
  du[1] = - β * S / N * I
  du[2] = β * S / N * I - σ * E
  du[3] = σ * E - γ * I
  du[4] = γ * I
end

# Parameters
γ = 1.0/18.0
σ = 1.0/5.2
R0 = 2.5
p = [γ, σ, R0]

I0 = 1.0e-7
E0 = 4.0 * I0
S0 = 1.0 - E0 - I0
R0 = 0.0

u0 = [S0, E0, I0, R0]

T = 18.0 * 30.0
tspan = (0.0, T)

# Define and solve problem
prob = ODEProblem(SIRE!, u0, tspan, p)
sol = solve(prob)

