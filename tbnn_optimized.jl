using DifferentialEquations, DiffEqFlux
using Flux
using StaticArrays

function tbnn(σ, D, model_weights, model_univ, t)
    #λ = zeros(SVector{9})
    λ = zeros(9) # must be mutable

    # Compute elements of the tensor basis
    I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
    T2 = σ
    T3 = D
    T4 = σ * σ
    T5 = D * D
    T6 = σ * D + D * σ
    T7 = T4 * D + D * T4
    T8 = σ * T5 + T5 * σ
    T9 = T4 * T5 + T5 * T4

    # Compute the integrity basis from scalar invariants. Traces. 
    λ[1] = tr(σ)
    λ[2] = tr(T4)
    λ[3] = tr(T5)
    λ[4] = tr(σ*σ*σ)
    λ[5] = tr(D*D*D)
    λ[6] = tr(T6)
    λ[7] = tr(T7)
    λ[8] = tr(T8)
    λ[9] = tr(T9)

    # Run the integrity basis through a neural network
    #model_inputs = [λ1; λ2; λ3; λ4; λ5; λ6; λ7; λ8; λ9]   # 9 x 1 matrix
    #g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(λ, model_weights)   # FIX NETWORK

    g = model_univ(λ, model_weights) 
    #println("g: ", g)

    # tst that this code is being executed. Plot should change. The code was indeed executing, 
    # and the solution did not change from its initial value. This must imply that the nonlinearity
    # has very little effect. 
    
    # Tensor combining layer
    F = g[1] .* I    +   g[2] .* σ    +   g[3] .* D    +   g[4] .* T4   +   g[5] .* T5 + 
        g[6] .* T6   +   g[7] .* T7   +   g[8] .* T8   +   g[9] .* T9
    
    #=
    # Capture data
    if dct[:captureG]
        coef = [t, g1,g2,g3,g4,g5,g6,g7,g8,g9]
        push!(tdnn_coefs, coef)
        trace = [t, λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8, λ9]
        push!(tdnn_traces, trace)
        F = [t, F11, F22, F33, F12, F13, F23]
        push!(tdnn_Fs, F)
    end
    =#

    return F
end

function dudt_univ(u, p, t, gradv, model_univ, n_weights, model_weights)
    # the parameters are [NN parameters, ODE parameters)

    η0, τ = @view p[end-1 : end]

    # Governing equations are for components of the stress tensor
    σ = SA_F32[u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    ∇v = SA_F32[0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
    D = 0.5 .* (∇v .+ transpose(∇v))

    T1 = (η0/τ) .* D 
    T2 = (transpose(∇v) * σ) + (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    # Change tbnn to read D
    F = tbnn(σ, D, model_weights, model_univ, t)

    du = -σ ./ τ + T1 + T2  - F ./ τ   # 9 equations (static matrix)
    #du = -σ  + T1 + T2     # 9 equations (static matrix)
end

function run()
    u = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
    t = 0.
    p = SA[1.,1.,1.]
    tspan = (0., 5.)
    σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]

    dct = Dict{Any, Any}()
    dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
    dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])
    dct[:model_weights] = [1,2,3] # FOR TESTING
    dct[:n_weights] = 3
    act = tanh
    hid = 32
    dct[:model_univ] = FastChain(FastDense(9, hid, act),
                       FastDense(hid, hid, act),
                       FastDense(hid, 9))
    p_model = initial_params(dct[:model_univ])
    n_weights = length(p_model)

    #dct[:model_weights] = 0.1 * rand(n_weights)
    dct[:model_weights] = zeros(n_weights)
    dct[:n_weights] = n_weights

    #n_weights::Int = dct[:n_weights]  
    #model_weights::Vector{Float32} = zeros(n_weights)  # CHECK
    model_univ = dct[:model_univ]
    model_weights = dct[:model_weights]

    # Extract from dictionary so that compiler can do type inference
    γ_protoc = dct[:γ_protoc]  # The type should now be correctly inferred on the LHS
    ω_protoc = dct[:ω_protoc]

    # mapping from time to ∂₁v₂ (i.e., an anomymous function)
    v21_protoc = (t) -> γ_protoc[1] * cos(ω_protoc[1]*t)

    dudt_uode(u, p, t) = dudt_univ(u, p, t, v21_protoc, model_univ, n_weights, model_weights)
    prob_giesekus_tbnn = ODEProblem(dudt_uode, σ0, tspan, p) # check on P
    @time sol_giesekus = solve(prob_giesekus_tbnn, Tsit5())
    @time sol_giesekus = solve(prob_giesekus_tbnn, Tsit5())
    @time sol_giesekus = solve(prob_giesekus_tbnn, Tsit5())
end

    #--------------------------------------

#----------------------------------------------------------------------
run()

