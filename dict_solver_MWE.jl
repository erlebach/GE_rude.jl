# ME to optimize Giesekus
using DifferentialEquations
using StaticArrays
using InteractiveUtils

function dudt_giesekus(u, p, t, gradv)
    # Destructure the parameters
	η0, τ, α = @view p[:]

    # Governing equations are for components of the 3x3 stress tensor
	σ = SA_F32[u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
	∇v = SA_F32[0. 0. 0. ; gradv[4](t) 0. 0. ; 0. 0. 0.]
	D = 0.5 .* (∇v .+ transpose(∇v))

	T1 = (η0/τ) .* D 
	T2 = (transpose(∇v) * σ) + (σ * ∇v)

	coef = α / (τ * η0)
	F = coef * (σ * σ)
	du = -σ / τ .+ T1 .+ T2  .- F .+ T2  # 9 equations (static matrix)
end

function run()
	du = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	t = 0.
	p = SA[1.,1.,1.]

	dct = Dict()
    dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
    dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])

	#@code_warntype dudt_giesekus(u, p, t, cos)

	tspan = (0., 5.)
	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]

	v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]
	v21_protoc = [ (t) -> 2. .* cos(3*t) for i in 1:8]

	#fct = (t) -> 2f0*cos(1.5*t)

	dudt(u,p,t) = dudt_giesekus(u, p, t, v21_protoc)
	prob_giesekus = ODEProblem(dudt, σ0, tspan, p)
	println("saveat=None, tspan=(0,5)")
	@time sol_giesekus = solve(prob_giesekus, Tsit5())
	@time sol_giesekus = solve(prob_giesekus, Tsit5())
	@time sol_giesekus = solve(prob_giesekus, Tsit5())
	println("saveat=0.2, tspan=(0,5)")
	@time sol_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.2)
	@time sol_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.2)
	println("saveat=0.02, tspan=(0,5)")
	@time sol_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.02)
	@time sol_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.02)
end

run()

