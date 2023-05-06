# Author: G. Erlebacher
# date: 2023-05-02
# Optimized Gisekus, minimal number of parameters and size

using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays
using BSON: @save, @load

VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 1
const max_nb_iter::Int32 = 100
start_at = 1 # Train from scratch

function open_files(files::Vector)
    # Open and close a file for writing. This should empty any 
    # existing file
    for fn in files
        file = open(fn, "w")
        close(file)
    end
end

open_files(["Giesekus_lambda.txt", "Giesekus_g.txt"])

# ==============================
function simple!(dσ, σ, p, t, gradv)
    α = p[1]
    dσ[1] = -α * σ[1] + gradv(t)
    return
end

tspan = (0., 1.)
p_simple = [2.]
dudt!(dσ, σ, p, t) = simple!(dσ, σ, p, t, gradv)
σ0 = 2.
dσ = 0.
prob = ODEProblem(dudt!, u0, tspan, p_simple)
solve_simple = solve(prob, Tsit5(), saveat=1.0);  # Do not show output of solve()
# =============================================

# Solve du/dt = -α u  where u is a 3x3 matrix
function solve3x3!(dσ, σ, p, t)
    # The dot is necessary to update dσ
    dσ .= -p[1] * σ
    return
end

tspan = (0., 2.)
p_simple = [4.]
σ0 = [1. 0. 0. ; 0. 2. 0. ; 3. 0. 0.]
dσ = zeros(3,3)
prob = ODEProblem(solve3x3!, σ0, tspan, p_simple)
solve_simple = solve(prob, Tsit5(), saveat=1.0);  # Do not show output of solve()
solve_simple.u

# =========================================
function dudt_giesekus_opt!(dσ, σ, p, t, gradv)
    # gradv is a
    # σ: 3x3 tensor
    # Destructure the parameters
    η0, τ, α = p
    ∇v = SA[0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
    D = 0.5 .* (∇v + transpose(∇v))
    T1 = (η0/τ) * D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)
    coef = α / (τ * η0)
    F = coef * (σ * σ)
    # the .= is necessary to updated dσ rather than overwrite it
    dσ .= -σ / τ .+ T1 .+ T2  .- F  # 9 equations (static matrix)
    return
end
# ===================
σ =  [0.0 0. 0. ; 0. 0. 0. ; 0. 0. 0.]
σ0 = [0.0 0. 0. ; 0. 0. 0. ; 0. 0. 0.]
p_giesekus = [1. 1. 1.]
t= 0.
dσ = [0.0 0. 0. ; 0. 0. 0. ; 0. 0. 0.]
v21_fct(t) = 1*cos(t)

#---------------------------------------
println("\n\n\n  GORDON EXPERIMENTS\n\n\n\n")
tspan = (0., 12.0)
dudt!(du,σ,p,t) = dudt_giesekus_opt!(du,σ,p,t,v21_fct);
prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
solve_giesekus = solve(prob_giesekus, Tsit5(), saveat=1.0);  # Do not show output of solve()
solve_giesekus.u

# Why is σ always zero? σ is not resset to σ + dσ . Why not?

# Rodas5 appears to use more complex solvers involving 
# forward differentiation as evidenced by more complex types for σ
#σ12_data = solve_giesekus.t

#-----------------------------------------------

# Define the simple shear deformation protocol

v12(t) = 0
v13(t) = 0
v22(t) = 0
v23(t) = 0
v31(t) = 0
v32(t) = 0
v33(t) = 0

# Iniitial conditions and time span
tspan = (0.00, 12.0)
tsave = range(tspan[1],tspan[2],length=50)
σ0 = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

# Build the protocols for different 'experiments'
ω = 1.0
v21_1(t) = 1*cos(ω*t)
v21_2(t) = 2*cos(ω*t)
v21_3(t) = 1*cos(ω*t/2)
v21_4(t) = 2*cos(ω*t/2)
v21_5(t) = 1*cos(2*ω*t)
v21_6(t) = 2*cos(2*ω*t)
v21_7(t) = 1*cos(ω*t/3)
v21_8(t) = 2*cos(ω*t/3)
gradv_1 = [v11,v12,v13,v21_1,v22,v23,v31,v32,v33]
gradv_2 = [v11,v12,v13,v21_2,v22,v23,v31,v32,v33]
gradv_3 = [v11,v12,v13,v21_3,v22,v23,v31,v32,v33]
gradv_4 = [v11,v12,v13,v21_4,v22,v23,v31,v32,v33]
gradv_5 = [v11,v12,v13,v21_5,v22,v23,v31,v32,v33]
gradv_6 = [v11,v12,v13,v21_6,v22,v23,v31,v32,v33]
gradv_7 = [v11,v12,v13,v21_7,v22,v23,v31,v32,v33]
gradv_8 = [v11,v12,v13,v21_8,v22,v23,v31,v32,v33]
protocols = [gradv_1, gradv_2, gradv_3, gradv_4, gradv_5, gradv_6, gradv_7, gradv_8]
tspans = [tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan]
tsaves = [tsave, tsave, tsave, tsave, tsave, tsave, tsave, tsave]

# Solve for the Giesekus model
η0 = 1
τ = 1
α = 0.8
p_giesekus = [η0,τ,α]
σ12_all = Any[]
t_all = Any[]
for k = range(1,max_nb_protocols)
    println("**************************************")
	#dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
	dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
    tspans[1] = (0.0, 12.0)
	prob_giesekus = ODEProblem(dudt!, σ0, tspans[k], p_giesekus)
	solve_giesekus = solve(prob_giesekus, Rodas4(), saveat=0.2) 
	σ12_data = solve_giesekus[4,:]
    print("==> ", σ12_data)
	push!(t_all, solve_giesekus.t)
	push!(σ12_all, σ12_data)
end

