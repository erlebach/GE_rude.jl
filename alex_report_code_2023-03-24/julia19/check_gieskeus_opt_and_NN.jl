# Date: 2023-05-07
# Author: Gordon Erlebacher, based of of RUDE Julia Implementation

# File to be included by software solving the Giesekus model with a 
# Tensor basis Neural Network

using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using BSON: @save, @load
using StableRNGs
using StaticArrays

function is_symmetric(D)
    n = norm(D - transpose(D))
    return n < 1.e-5
end

function mat3x3_to_vec6(D)
    d = zeros(6)
    d[1] = D[1,1]
    d[2] = D[2,2]
    d[3] = D[3,3]
    d[4] = D[1,2]
    d[5] = D[1,3]
    d[6] = D[2,3]
    return d
end

function vec6_to_mat3x3(v)
    D = SizedMatrix{3,3}(zeros(3,3))
    D[1,1] = v[1];   D[1,2] = v[4];   D[1,3] = v[5]
    D[2,1] = v[4];   D[2,2] = v[2];   D[2,3] = v[6]
    D[3,1] = v[5];   D[3,2] = v[6];   D[3,3] = v[3]
    return D
end

# This file is included in a file 
function dudt_giesekus_opt!(du, σ, p, t, gradv)
    η0, τ, α = p
    # gradv is an array of functions
    ∇v = SA[0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0]
    D = (∇v .+ transpose(∇v))  # necessary to produce same result as original RUDE
    T1 = (η0 / τ) .* D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)
    coef = α / (τ * η0)
    F = coef * (σ * σ)
    du .= -σ / τ .+ T1 .+ T2 .- F  # 9 equations (static matrix)
end

#---------------------------------------------------------------
function dudt_univ_opt!(dσ, σ, p, t, gradv, model_univ, model_weights)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1:end]
    ∇v = SizedMatrix{3,3}([0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0])
    D = (∇v .+ transpose(∇v))  # probably necessary to match original RUDE
    T1 = (η0 / τ) .* D
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
    dσ .= (-σ / τ) .+ T1 .+ T2 .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    λ = similar(σ, (9,))

    # Compute elements of the tensor basis
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
    λ[5] = tr(T5 * D)
    λ[6] = T4[1,1]*T5[1,1] + T4[2,2]*T5[2,2] + T4[3,3]*T5[3,3] +
        2.0*(T4[1,2]*T5[1,2] + T4[1,3]*T5[1,3] + T4[2,3]*T5[2,3])
    λ[7] = tr(T7) * 0.5
    λ[8] = tr(T8) * 0.5
    λ[9] = tr(T6) * 0.5

    println("(opt) enter tbnn, λ = ", λ)

    g = re(model_weights)(λ)

    println("(opt) exit tbnn, g= ", g)

    F = g[1] .* I + g[2] .* σ + g[3] .* D + g[4] .* T4 + g[5] .* T5 +
        g[6] .* T6 + g[7] .* T7 + g[8] .* T8 + g[9] .* T9
end

function NeuralNetwork(; nb_in=1, nb_out=1, layer_size=8, nb_hid_layers=1)
    # Note use of Any[] since the Dense layers are of different types
    layers = Any[Flux.Dense(nb_in => layer_size, tanh)]
    push!(layers, [Flux.Dense(layer_size => layer_size, tanh) for i in 1:nb_hid_layers-1]...)
    push!(layers, Flux.Dense(layer_size => nb_out))
    return Chain(layers...)
end

# =======================================================
# CHECK giesekus and the NN with pre-specified input
# Execute dudt_giesekus with random I.C. and compare to my new code.
#---------------------------------------------------------------

function check_giesekus_opt()
    rng = StableRNG(1234)
    u0 = rand(rng, 6)
    σ0 = rand(rng, 3, 3)
    σ0[1,1], σ0[2,2], σ0[3,3], σ0[1,2], σ0[1,3], σ0[2,3] = [u0[i] for i in 1:6]
    σ0[2,1] = σ0[1,2]
    σ0[3,1] = σ0[1,3]
    σ0[3,2] = σ0[2,3]
    p = (1., 1., 0.8)
    println("p: ", p)
    t = 3.
    gradv = Any[t -> 0. for i in 1:3 for j in 1:3]
    gradv[2,1] = t -> cos(t)
    gradv = reshape(gradv, (3,3))
    # println(size(gradv))
    du = similar(σ0, 3, 3)
    dd = dudt_giesekus_opt!(du, σ0, p, t, gradv)
    println("\ncheck_giesekus_opt: \n", du)
    return du 
end

function check_giesekus_opt_NN() 
    global re
    rng = StableRNG(1234)
    u0 = rand(rng, 6)
    σ0 = rand(rng, 3, 3)
    σ0[1,1], σ0[2,2], σ0[3,3], σ0[1,2], σ0[1,3], σ0[2,3] = [u0[i] for i in 1:6]
    σ0[2,1] = σ0[1,2]
    σ0[3,1] = σ0[1,3]
    σ0[3,2] = σ0[2,3]

    p_giesekus = [1., 1.]  # Must be a list
    t = 3.
    gradv = [t -> 0.  t -> 0.  t -> 0.; t -> cos(t)  t -> 0.  t -> 0.; t -> 0.  t -> 0.  t -> 0.]
    println("1. gradv: ", [grad(t) for grad in gradv])
    du = similar(σ0, 3, 3)

    model_univ = NeuralNetwork(; nb_in=9, nb_out=9, layer_size=8, nb_hid_layers=0)
    # println("===========")
    # println(model_univ)
    # println("===========")
    p_model, re = Flux.destructure(model_univ)
    # p_model = zeros(size(p_model))  # zero weights in a single list
    rng = StableRNG(4321)
    p_model = 0.01 .* (-1. .+ 2 .* rand(rng, size(p_model)[1]))  # zero weights in a single list

    # @show p_model
    p = [p_model; p_giesekus]
    model_weights = p_model
    println("len p: ", p|>length)
                                             #           <<<  re  >>>>
    dudt_univ_opt!(du, σ0, p, t, gradv, model_univ, model_weights)
    # @show model_weights
    println("du: ", du)
    println("du symmetric? ", is_symmetric(du))
    println("\ncheck_giesekus_opt_NN: ")
    display(du)
    println("==================================================")
    return du
end
#---------------------------------------------------------------
check_giesekus_opt();
duNN = check_giesekus_opt_NN();
println("================================================")

### WHY IS the output of giesekus_opt_NN independent of NN structure? 