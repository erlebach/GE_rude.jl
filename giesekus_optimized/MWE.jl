# ME to optimize Giesekus
using DifferentialEquations
using StaticArrays
using InteractiveUtils

function dudt_giesekus(u, p, t, gradv)
    # Destructure the parameters
	η0, τ, α = p

	@show t

    # Governing equations are for components of the 3x3 stress tensor
	σ = @SMatrix [u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
	∇v = @SMatrix [0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
	D = 0.5 .* (∇v + transpose(∇v))

	T1 = (η0/τ) .* D 
	T2 = (transpose(∇v) * σ) + (σ * ∇v)

	coef = α / (τ * η0)
	F = coef * (σ * σ)
	du = -σ / τ + T1 + T2  - F  # 9 equations (static matrix)
end

function run()
	du = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u0 = @SMatrix [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	t = 0.
	p = SA[1.,1.,1.]

	dct = Dict{Any, Any}()
    dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
    dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])

	γ_protoc = dct[:γ_protoc]  # The type should now be correctly inferred on the LHS
	ω_protoc = dct[:ω_protoc]

	dudt_giesekus(u, p, t, cos)

	tspan = (0., 5.)
	σ0 = @SMatrix [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]

	# Memory allocation is 12k per call to solve(). WHY? 
	v21_protoc = (t) -> γ_protoc[1] * cos(ω_protoc[1]*t)

	dudt(u,p,t) = dudt_giesekus(u, p, t, v21_protoc)
	prob_giesekus = ODEProblem(dudt, σ0, tspan, p)

	#sol_giesekus = solve(prob_giesekus, Tsit5())
	sol_giesekus = solve(prob_giesekus, Rodas4())
end

run()

