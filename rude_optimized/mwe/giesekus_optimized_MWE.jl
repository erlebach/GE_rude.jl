# ME to optimize Giesekus
using DifferentialEquations
using StaticArrays
using InteractiveUtils

function dudt_giesekus_opt(u, p, t, gradv)
    # Destructure the parameters
	#println("==> enter dudt_gie...")
	η0, τ, α = p

	#println("dudt_giesekus_opt, u: ", typeof(u))

    # Governing equations are for components of the 3x3 stress tensor
	#σ = SA_F32[u[1] u[2] 0.; u[4] 0. 0.; 0. 0. u[3]]
	#σ = SA_F32[u[1] u[2] u[3]; u[4] u[5] u[6]; u[7] u[8] u[9]]
	#println("dudt..., u= ", u)
	σ = u
	#println("dudt_giesekus_opt, σ: ", typeof(σ))

	#println("dudt_giesekus_opt, σ: ", typeof(σ))

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
	#println("gradv: ", typeof(gradv))
	#println("t: ", typeof(t))
	# Why would t become ForwardDiffDual all of a sudden?
	#t: ForwardDiff.Dual{ForwardDiff.Tag{SciMLBase.TimeDerivativeWrapper{ODEFunction{false,SciMLBase.AutoSpecialize,…}, StaticArraysCore.SMatrix{3, 3, Float64, 9}, StaticArraysCore.SVector{3, Float32}}, Float32}, Float32, 1}
	∇v = @SMatrix [0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
	#∇v = @SMatrix [0. 0. 0. ; 0. 0. 0. ; 0. 0. 0.] # ok
	println("∇v= ", ∇v, ∇v |> typeof)
	D = 0.5 .* (∇v + transpose(∇v))
	#println("D: ", typeof(D))

	T1 = (η0/τ) .* D 
	T2 = (transpose(∇v) * σ) + (σ * ∇v)

	coef = α / (τ * η0)
	F = coef * (σ * σ)
	du = -σ / τ + T1 + T2  - F  # 9 equations (static matrix)
	#println("du: ", typeof(du))
	#println("==> Exit dudt_gi...opt")
	# Return a regular matrix (or should I return a static array)? 
	#duu = convert(Vector{Float64}, du)  # convert 2D array to 1D array. Should convert to 2D array. 
	return du
end

#=
function run()
	du = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	u0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
	t = 0.
	p = SA[1.,1.,1.]

	dct = Dict{Any, Any}()
    dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
    dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])

	γ_protoc = dct[:γ_protoc]  # The type should now be correctly inferred on the LHS
	ω_protoc = dct[:ω_protoc]

	#@code_warntype dudt_giesekus(u, p, t, cos)
	dudt_giesekus(u, p, t, cos)

	tspan = (0., 5.)
	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]

	#v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]
	#v21_protoc = [ (t) -> 2. .* cos(3*t) for i in 1:8]

	# Memory allocation is 12k per call to solve(). WHY? 
	#v21_protoc = (t) -> dct[:γ_protoc][1]*cos(dct[:ω_protoc][1]*t)
	v21_protoc = (t) -> γ_protoc[1] * cos(ω_protoc[1]*t)

	#fct = (t) -> 2f0*cos(1.5*t)  # Very efficient. No memory allocation

	dudt(u,p,t) = dudt_giesekus(u, p, t, v21_protoc)
	prob_giesekus = ODEProblem(dudt, σ0, tspan, p)

	sol_giesekus = solve(prob_giesekus, Tsit5())
	sol_giesekus = solve(prob_giesekus, Tsit5())

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
=#
