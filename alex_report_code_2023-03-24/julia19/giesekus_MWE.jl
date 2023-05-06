
#using Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays

VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 1
const max_nb_iter::Int32 = 100
start_at = 1 # Train from scratch

function dudt_giesekus_opt!(du, u, p, t, gradv)
    # Destructure the parameters
    η0, τ, α = p

    # Governing equations are for components of the 3x3 stress tensor
    σ = u

    println("gradv: ", gradv)  # gradv is an array
    ∇v = @SMatrix [0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]   # ERROR
    # ∇v = @SMatrix [0. 0. 0. ; 0.  0. 0. ; 0. 0. 0.]   # ERROR
    #=
    #∇v = @SMatrix [0. 0. 0. ; 0. 0. 0. ; 0. 0. 0.] # ok
    
    println("∇v= ", ∇v, ∇v |> typeof)
    D = 0.5 .* (∇v + transpose(∇v))

    T1 = (η0/τ) .* D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)

    coef = α / (τ * η0)
    F = coef * (σ * σ)
    #du = -σ / τ + T1 + T2  - F  # 9 equations (static matrix)
    =#
    du = -σ / τ 
    #return du
end

# Iniitial conditions and time span
tspan = (0.0f0, 12f0)
σ0 = [0f0,0f0,0f0,0f0,0f0,0f0]

# Solve for the Giesekus model
p_giesekus = [1, 1, 0.8]
k = 1

v21_protoc = (t) -> cos(t)
gradv = [0., v21_protoc, 0., 0.]
println("v21_proc(3.): ", v21_protoc(3.))

# For some reason, function wants to call [k]
# Last argument of dudt_giesekus_opt! is a function
dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,gradv[2])
prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
solve_giesekus = solve(prob_giesekus, Rodas4(), saveat=0.2)  ### ERROR
