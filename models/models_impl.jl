function gamma_dot(t, γ₀, ω)
    return γ₀ * ω * cos(ω * t)
    #return γ₀ .* ω .* cos.(ω .* t)
end

mutable struct Model
    λ::Float64
    G::Float64
    γ₀::Float64
    ω::Float64
    ϵ::Float64
    α::Float64
end

function ODE_UCM!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω = p
    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    du[1] = -(1. / λ) * σ11 + 2. * γ̇ * σ12
    du[2] = -(1. / λ) * σ22
    du[3] = -(1. / λ) * σ33
    du[4] = -(1. / λ) * σ12 + γ̇ * σ22 + G * γ̇
    return du
end # module

function ODE_PTT!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω, ϵ = p
exp(ϵ * trace / G) / λ    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    trace = σ11 + σ22 + σ33
    fPTTλ = exp(ϵ * trace / G) / λ
    du[1] = -fPTTλ * σ11 + 2. * γ̇ * σ12
    du[2] = -fPTTλ * σ22
    du[3] = -fPTTλ * σ33
    du[4] = -fPTTλ * σ12 + γ̇ * σ22 + G * γ̇
    return du
end # module

# When is (ϵ*trace/G) == O(1)? 
# Answer: when $trace = G / ϵ)
# Trace approx 0.1, G = 1 ==> ϵ = G / trace = 10

function ODE_G!(du, u, p, t, gamma_dot)
    λ, G, γ₀, ω, α = p
    γ̇ = gamma_dot(t, γ₀, ω)
    σ11, σ22, σ33, σ12 = u
    coef = α / (λ * G) 
    du[1] = -σ11 / λ - coef * (σ11^2 + σ12^2) + 2. * γ̇ * σ12 
    du[2] = -σ22 / λ - coef * (σ22^2 + σ12^2)
    du[3] = -σ33 / λ - coef * σ33^2
    du[4] = -σ12 / λ - coef * ((σ11 + σ22) * σ12) + γ̇ * σ22 + G * γ̇
    return du
end # module


function calculate_and_plot(u0, Δt, tspan)
    dct = Dict(
        :λ => model_G.λ,
        :G => model_G.G,
        :γ₀ => model_G.γ₀,
        :ω => model_G.ω,
        :ϵ => model_G.ϵ,
        :α => model_G.α,
    )

    # Closure
    ODE_UCM_!(u, du, p, t) = ODE_UCM!(u, du, p, t, gamma_dot)
    ODE_PTT_!(u, du, p, t) = ODE_PTT!(u, du, p, t, gamma_dot)
    ODE_G_!(u, du, p, t)   = ODE_G!(u, du, p, t, gamma_dot)

    function solve_ODEs(odes, u0, tspan, p, Δt)
        prob = ODEProblem(odes, u0, tspan, p)
        solution = solve(prob, Tsit5(), abstol=1.e-7, reltol=1.e-7, saveat=Δt)
        dct = Dict()
        dct[:t] = solution.t
        dct[:X] = Array(solution)
        grid = (:xy, :olivedrab, :dot, 1, 0.9)
        dct[:plot] = plot(dct[:t], dct[:X]', grid=grid, legend=:topright)
        # Each curve is the soluiton to one equation. The bottom curve is σ12
        # dct[:X]' is of dimension (4, length(t))
        return dct
    end

    dct_UCM = solve_ODEs(ODE_UCM_!, u0, tspan, p_UCM, Δt)
    dct_PTT = solve_ODEs(ODE_PTT_!, u0, tspan, p_PTT, Δt)
    dct_G   = solve_ODEs(ODE_G_!, u0, tspan, p_G, Δt)
    return dct, dct_UCM, dct_PTT, dct_G
end
