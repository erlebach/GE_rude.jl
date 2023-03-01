using DifferentialEquations
using StaticArrays
using InteractiveUtils

function tst()
	a = transpose(SA[1,2,3])
	b = SA[1, 2, 3]
	return a*b
end

function dudt_giesekus(u, p, t, gradv)
    # Destructure the parameters
    η0 = p[1]
    τ  = p[2]  
    α  = p[3]

	# Solving 3x3 system  (as opposed to 6x1 system previously)

    # Governing equations are for components of the stress tensor
    #σ11,σ22,σ33,σ12,σ13,σ23 = u
	σ = SA_F32[u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]

    # Specify the velocity gradient tensor
    # ∇v  ∂vᵢ / ∂xⱼ v12: partial derivative of the 1st component 
    # of v(x1, x2, x3) with respect to the 3rd component of x
	#v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv      # only v21 is non-zero for Alex

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
	#∇v = SA_F32[gradv[1](t)  gradv[2](t)  gradv[3](t); gradv[4](t)  gradv[5](t)  gradv[6](t); gradv[7](t)  gradv[8](t)  gradv[9](t)]
	∇v = SA_F32[0. 0. 0. ; gradv[4](t) 0. 0. ; 0. 0. 0.]
	D = 0.5 .* (∇v .+ transpose(∇v))

	T1 = (η0/τ) .* D 
	T2 = (transpose(∇v) * σ) + (σ * ∇v)

	coef = α / (τ * η0)
	F = coef * (σ * σ)

    # The model differential equations
	# Why is memory allocated? Must be the RHS. Redo this with ModelingToolkit. 
	du = -σ/τ .+ T1 .+ T2  .- F .+ T2  # 9 equations (static matrix)
	# du is 3x3 symmetric matrix. Howver, the speed is 2x previous implementation, and easier to read. 
end

const v11(t) = 0
const v12(t) = 0
const v13(t) = 0
const v22(t) = 0
const v23(t) = 0
const v31(t) = 0
const v32(t) = 0
const v33(t) = 0

p = SA[1., 1., 1.]

const dct = Dict()
dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])

# Iniitial conditions and time span
#const v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]
const v21_protoc = (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t)
const gradv = [v11,v12,v13,  v21_protoc, v22,v23,v31,v32,v33] 

du = [0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
u = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]
tspan = (0., 5.)

const t = 0.

println("dudt_giesekus")
dudt_giesekus(u, p, t, gradv)

### @time dudt_giesekus(u, p, t, gradv)
### @time dudt_giesekus(u, p, t, gradv)
### 
### 
### dudt(u,p,t) = dudt_giesekus(u,p,t,gradv)
### prob_giesekus = ODEProblem(dudt, σ0, tspan, p)
### @time sol_giesekus = solve(prob_giesekus,Tsit5())
### @time sol_giesekus = solve(prob_giesekus,Tsit5())
### @time sol_giesekus = solve(prob_giesekus,Tsit5())
### 
