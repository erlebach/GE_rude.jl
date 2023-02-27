# Three model implementations: UCM, PTT, GE
# Objective: explore effect of parameters and numerical solvers on solution

using DifferentialEquations
using Plots
include("models_impl.jl")
λ = 0.1
G = 1.0
ω = 1.
γ₀ = 20.
ϵ = 0.1   # Set to zero, and the solution of PTT matches UCM
α = 0.1 # Set to zero, and the solution of G matches UCM
model_UCM =  Model(λ, G, γ₀, ω, ϵ, α)
model_PTT =  Model(λ, G, γ₀, ω, ϵ, α)
model_G   =  Model(λ, G, γ₀, ω, ϵ, α)
f_UCM(m::Model) = [m.λ, m.G, m.γ₀, m.ω]  # function
f_PTT(m::Model) = [m.λ, m.G, m.γ₀, m.ω, m.ϵ]
f_G(m::Model) = [m.λ, m.G, m.γ₀, m.ω, m.α]
p_UCM = f_UCM(model_UCM)   # list
p_PTT = f_PTT(model_PTT)
p_G   = f_G(model_G)

Ncycles = 10
nb_per_cycle = 20 
T = Ncycles * (2. * π / model_UCM.ω)
tspan = (0., T)
u0 = [0., 0., 0., 0.] # (page 3 of Sachin project description [SPD22])
Δt = T / (Ncycles * nb_per_cycle)

dct, dct_UCM, dct_PTT, dct_G = calculate_and_plot(u0, Δt, tspan)
a = plot!(dct_UCM[:plot], title="UCM")
b = plot!(dct_PTT[:plot], title="PTT")
c = plot!(dct_G[:plot], title="G")
plot(a, b, c, layout=(3,1))