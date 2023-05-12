# Date: 2023-05-07
# Author: Gordon Erlebacher, based of of RUDE Julia Implementation

# File to be included by software solving the Giesekus model with a 
# Tensor basis Neural Network

# This file is included in a file 
function dudt_giesekus_opt!(du, σ, p, t, gradv)
    η0, τ, α = p
    # gradv is an array of functions
    # ∇v = SA[g(t) for g in gradv]
    # of   ∇v = SA[0. gradv[2,1](t) 0. ; 0. 0. 0. ; 0. 0. 0.]
    ∇v = SA[0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0]
    # println("∇v: ", ∇v)
    #∇v = [gradv[i](t) for i in 1:length(gradv)]
    # D = 0.5 .* (∇v .+ transpose(∇v))  # orig
    D = (∇v .+ transpose(∇v))  # necessary to produce same result as original RUDE
    T1 = (η0 / τ) * D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)
    coef = α / (τ * η0)
    F = coef * (σ * σ)
    du .= -σ / τ .+ T1 .+ T2 .- F  # 9 equations (static matrix)
end

#---------------------------------------------------------------


fct_giesekus = function (tspan, tsave, p, σ0, protocol)
    # p: parameters for Giesekus base model
    dudt!(du, u, p, t) = dudt_giesekus_opt!(du, u, p, t, protocol)
    prob = ODEProblem(dudt!, σ0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tsave)
end

fct_ude = function (tspan, tsave, θ, p, σ0, protocol, model_univ, model_weights)
    dudt_ude!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocol, model_univ, model_weights)
    # Solve the UDE 
    prob_ude = ODEProblem(dudt_ude!, σ0, tspan, [θ; p])
    sol_ude = solve(prob_ude, Tsit5(), abstol=1e-7, reltol=1e-6, saveat=tsave)
end

function dudt_univ_opt!(dσ, σ, p, t, gradv, model_univ, model_weights)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1:end]
    ∇v = SizedMatrix{3,3}([0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0])
    # D = 0.5 .* (∇v .+ transpose(∇v))
    D = (∇v .+ transpose(∇v))  # probably necessary to match original RUDE
    T1 = (η0 / τ) .* D
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
    dσ .= (-σ ./ τ) .+ T1 .+ T2 .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    λ = similar(σ, (9,))

    # Compute elements of the tensor basis
    #I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]  # Generates an error. Already reported.
    I = SizedMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
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
    λ[4] = tr(T4 * σ)
    #λ[5] = tr(D * D * D)
    λ[5] = tr(T5 * D)
    λ[6] = T4[1,1]*T5[1,1] + T4[2,2]*T5[2,2] + T4[3,3]*T5[3,3] +
        2.0*(T4[1,2]*T5[1,2] + T4[1,3]*T5[1,3] + T4[2,3]*T5[2,3])
    λ[7] = tr(T7) * 0.5
    λ[8] = tr(T8) * 0.5
    λ[9] = tr(T6) * 0.5

    g = re(model_weights)(λ)

    F = g[1] .* I + g[2] .* σ + g[3] .* D + g[4] .* T4 + g[5] .* T5 +
        g[6] .* T6 + g[7] .* T7 + g[8] .* T8 + g[9] .* T9
end

# ====================

function setup_protocols(n_protocol, v21_fct, tspan)
    #function setup_protocols(ω) 
    #= 
    Define the simple shear deformation protocols. 
    TODO Generalize to handle different number and types of protocols 
    =#
    # Any is necessary since I will be inserting a function of different type 
    # in one or more of the array elements
    println("inside setup_protocols")
    vij_fct = SizedMatrix{3,3,Any}(t -> 0.0 for i in 1:3, j in 1:3)

    # Build the protocols for different 'experiments'
    protocols = Vector{Any}()  # each element is 3x3 matrix of functions of shear ∂ij(v) (w/ or w/o time-derivative?)

    println("====> ")
    @show n_protocol
    for i in 1:n_protocol
        push!(protocols, copy(vij_fct))
        protocols[i][2, 1] = v21_fct[i]
    end

    # Iniitial conditions and time span
    # tspan = (0.0f0, 12f0)
    tsave = range(tspan[1], tspan[2], length=50)
    tspans = [tspan for i in 1:n_protocol]
    tsaves = [tsave for i in 1:n_protocol]
    return protocols, tspans, tsaves
end

function solve_giesekus_protocols(protocols, tspans, p, σ0, max_nb_protocols; dudt_sys=dudt_giesekus_opt!)
    # Ideally, the function should be self-contained. If it is, it can be ported anywhere.
    # Solve for the Giesekus model
    @show max_nb_protocols  # 8
    σ12_all = Any[]
    t_all = Any[]
    for k = range(1, max_nb_protocols)
        println("giesekus solve, k= ", k)
        dudt!(du, u, p, t) = dudt_sys(du, u, p, t, protocols[k])  # the argument beyond t depends on the model
        prob_giesekus = ODEProblem{true,SciMLBase.FullSpecialize}(dudt!, σ0, tspans[k], p)
        # solve_giesekus = solve(prob_giesekus, Rodas4(), saveat=0.2) 
        # Stack error with Rodas4 and Tsit5
        solve_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.2)
        σ12_data = solve_giesekus[4, :]
        push!(t_all, solve_giesekus.t)
        push!(σ12_all, σ12_data)
    end
    return t_all, σ12_all
end

function NeuralNetwork(; nb_in=1, nb_out=1, layer_size=8, nb_hid_layers=1)
    # Note use of Any[] since the Dense layers are of different types
    layers = Any[Flux.Dense(nb_in => layer_size, tanh)]
    push!(layers, [Flux.Dense(layer_size => layer_size, tanh) for i in 1:nb_hid_layers-1]...)
    push!(layers, Flux.Dense(layer_size => nb_out))
    return Chain(layers...)
end

#*****************************************************************************************
# Helpers to solve the UODE 
#*****************************************************************************************

function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, model_univ, model_weights)
    # trajectories is an integer
    dudt_protocol!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[1], model_univ, model_weights)
    prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

    # Remake the problem for different protocols
    function prob_func(prob, i, repeat)
        dudt_remade!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[i], model_univ, model_weights)
        remake(prob, f=dudt_remade!, tspan=tspans[i])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories,
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)  ### ERROR
end

function loss_univ(θ, protocols, tspans, σ0, σ12_all, nb_trajectories, model_univ, model_weights)
    # length(protocols) = 1, then 2, ...
    # println("loss_univ, length protocols: ", length(protocols))
    # nb_trajectories is either 1, or 2, or 3, ..., length(nb_trajectories) = 1
    # println("loss_univ, length nb_trajetories: $(length(nb_trajectories)), $nb_trajectories") # always 1 trajectory
    loss = 0
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, nb_trajectories, model_univ, model_weights)
    for k = range(1, nb_trajectories, step=1)
        σ12_pred = results[k][4, :]
        σ12_data = σ12_all[k]
        loss += sum(abs2, σ12_pred - σ12_data)
    end
    loss += 0.01 * norm(θ, 1)
    return loss
end

function solve_UODE(θi, max_nb_protocols, p_system, tspans, σ0, σ12_all, callback, model_univ, loss_univ, model_weights, max_nb_iter)
    # Continutation training loop
    adtype = Optimization.AutoZygote()
    k = 1
    # iter = 0
    print("p_system: ", p_system)
    results_univ = []
    for k = range(1, max_nb_protocols)
        println("k= ", k)
        loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, model_univ, model_weights)
        cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all, k)
        optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)
        result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)
        push!(results_univ, result_univ)
        @show result_univ.u
        @show length(result_univ.u)
        θi = result_univ.u
    end
    return results_univ
end
