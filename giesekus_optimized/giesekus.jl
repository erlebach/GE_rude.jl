using DifferentialEquations


function dudt_giesekus!(du, u, p, t, gradv)
    # Destructure the parameters
    η0 = p[1]
    τ = p[2]  # \lambda
    α = p[3]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    # ∇v  ∂vᵢ / ∂xⱼ v12: partial derivative of the 1st component 
    # of v(x1, x2, x3) with respect to the 3rd component of x
	v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv      # only v21 is non-zero for Alex

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)  
    γd22 = 2*v22(t) 
    γd33 = 2*v33(t)
    γd12 = v12(t) + v21(t)  # = v21
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)
    ω12 = v12(t) - v21(t)  # = -v21
    ω13 = v13(t) - v31(t)  
    ω23 = v23(t) - v32(t)

    # Define F for the Giesekus model
    F11 = -τ*(σ11*γd11 + σ12*γd12 + σ13*γd13) + (α*τ/η0)*(σ11^2 + σ12^2 + σ13^2)
    F22 = -τ*(σ12*γd12 + σ22*γd22 + σ23*γd23) + (α*τ/η0)*(σ12^2 + σ22^2 + σ23^2)
    F33 = -τ*(σ13*γd13 + σ23*γd23 + σ33*γd33) + (α*τ/η0)*(σ13^2 + σ23^2 + σ33^2)
    F12 = (-τ*(σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23)/2
    + (α*τ/η0)*(σ11*σ12 + σ12*σ22 + σ13*σ23))
    F13 = (-τ*(σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33)/2
    + (α*τ/η0)*(σ11*σ13 + σ12*σ23 + σ13*σ33))
    F23 = (-τ*(σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33)/2
    + (α*τ/η0)*(σ12*σ13 + σ22*σ23 + σ23*σ33))

    ##
    # The model differential equations
	# Why is memory allocated? Must be the RHS. Redo this with ModelingToolkit. 
	# Limit to four equations for further speed (perhaps not)
    du[1] = η0*γd11/τ - σ11/τ - (ω12*σ12 + ω13*σ13) - F11/τ
    du[2] = η0*γd22/τ - σ22/τ - (ω23*σ23 - ω12*σ12) - F22/τ
    du[3] = η0*γd33/τ - σ33/τ + (ω13*σ13 + ω23*σ23) - F33/τ
    du[4] = η0*γd12/τ - σ12/τ - (ω12*σ22 + ω13*σ23 - σ11*ω12 + σ13*ω23)/2 - F12/τ
    du[5] = η0*γd13/τ - σ13/τ - (ω12*σ23 + ω13*σ33 - σ11*ω13 - σ12*ω23)/2 - F13/τ
    du[6] = η0*γd23/τ - σ23/τ - (ω23*σ33 - ω12*σ13 - σ12*ω13 - σ22*ω23)/2 - F23/τ
	nothing
    ##
end

v11(t) = 0
v12(t) = 0
v13(t) = 0
v22(t) = 0
v23(t) = 0
v31(t) = 0
v32(t) = 0
v33(t) = 0

p = [1., 1., 1.]

const dct = Dict()
dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])

# Iniitial conditions and time span
const v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]
const gradv = [v11,v12,v13,  v21_protoc[1], v22,v23,v31,v32,v33] 

const du = [0., 0., 0., 0., 0., 0.]
const u = [0., 0., 0., 0., 0., 0.]

@time dudt_giesekus!(du, u, p, 0., gradv)
@time dudt_giesekus!(du, u, p, 0., gradv)
#@time dudt_giesekus!(du, u, p, 0., gradv)
#@time dudt_giesekus!(du, u, p, 0., gradv)
#@time dudt_giesekus!(du, u, p, 0., gradv)
#@time dudt_giesekus!(du, u, p, 0., gradv)
#
tspan = (0., 5.)
const σ0 = [0., 0., 0., 0., 0., 0.]
#
dudt!(du,u,p,t) = dudt_giesekus!(du,u,p,t,gradv)
prob_giesekus = ODEProblem(dudt!, σ0, tspan, p)
println("saveat=None, tspan=(0,5)")
@time sol_giesekus = solve(prob_giesekus,Tsit5())
@time sol_giesekus = solve(prob_giesekus,Tsit5())
@time sol_giesekus = solve(prob_giesekus,Tsit5())
println("saveat=0.2, tspan=(0,5)")
@time sol_giesekus = solve(prob_giesekus,Tsit5(), saveat=0.2)
@time sol_giesekus = solve(prob_giesekus,Tsit5(), saveat=0.2)
println("saveat=0.02, tspan=(0,5)")
@time sol_giesekus = solve(prob_giesekus,Tsit5(), saveat=0.02)
@time sol_giesekus = solve(prob_giesekus,Tsit5(), saveat=0.02)
#
#dct[:saveat] = 0.0005
#@time sol_giesekus = solve(prob_giesekus,Tsit5(),saveat=dct[:saveat])
#@time sol_giesekus = solve(prob_giesekus,Tsit5(),saveat=dct[:saveat])
#@time sol_giesekus = solve(prob_giesekus,Tsit5(),saveat=dct[:saveat])
