using DifferentialEquations

# change to tbnn!, add a first argument of preallocated T
function tbnn(σ, γd, model_weights, model_univ, t)
    # Tensor basis neural network (TBNN)
	# 2023-02-27: Added t to the argument list ot allow me to capture (g1->g9,t) 
    # Unpack the inputs
    σ11,σ22,σ33,σ12,σ13,σ23 = σ
    γd11,γd22,γd33,γd12,γd13,γd23 = γd

    # Compute elements of the tensor basis
    # T1 = I, T2 = σ, T3 = γd
    # T4 = σ⋅σ
    T4_11 = σ11^2 + σ12^2 + σ13^2
    T4_22 = σ12^2 + σ22^2 + σ23^2
    T4_33 = σ13^2 + σ23^2 + σ33^2
    T4_12 = σ11*σ12 + σ12*σ22 + σ13*σ23
    T4_13 = σ11*σ13 + σ12*σ23 + σ13*σ33
    T4_23 = σ12*σ13 + σ22*σ23 + σ23*σ33

    # T5 = γd⋅γd
	# Why are memory allocations zero? 
    T5_11 = γd11^2 + γd12^2 + γd13^2
    T5_22 = γd12^2 + γd22^2 + γd23^2
    T5_33 = γd13^2 + γd23^2 + γd33^2
    T5_12 = γd11*γd12 + γd12*γd22 + γd13*γd23
    T5_13 = γd11*γd13 + γd12*γd23 + γd13*γd33
    T5_23 = γd12*γd13 + γd22*γd23 + γd23*γd33

    # T6 = σ⋅γd + γd⋅σ
    T6_11 = 2*(σ11*γd11 + σ12*γd12 + σ13*γd13)
    T6_22 = 2*(σ12*γd12 + σ22*γd22 + σ23*γd23)
    T6_33 = 2*(σ13*γd13 + σ23*γd23 + σ33*γd33)
    T6_12 = σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23
    T6_13 = σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33
    T6_23 = σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33

    # T7 = σ⋅σ⋅γd + γd⋅σ⋅σ
    T7_11 = 2*(T4_11*γd11 + T4_12*γd12 + T4_13*γd13)
    T7_22 = 2*(T4_12*γd12 + T4_22*γd22 + T4_23*γd23)
    T7_33 = 2*(T4_13*γd13 + T4_23*γd23 + T4_33*γd33)
    T7_12 = T4_11*γd12 + T4_12*γd22 + T4_13*γd23 + γd11*T4_12 + γd12*T4_22 + γd13*T4_23
    T7_13 = T4_11*γd13 + T4_12*γd23 + T4_13*γd33 + γd11*T4_13 + γd12*T4_23 + γd13*T4_33
    T7_23 = T4_12*γd13 + T4_22*γd23 + T4_23*γd33 + γd12*T4_13 + γd22*T4_23 + γd23*T4_33

    # T8 = σ⋅γd⋅γd + γd⋅γd⋅σ
    T8_11 = 2*(σ11*T5_11 + σ12*T5_12 + σ13*T5_13)
    T8_22 = 2*(σ12*T5_12 + σ22*T5_22 + σ23*T5_23)
    T8_33 = 2*(σ13*T5_13 + σ23*T5_23 + σ33*T5_33)
    T8_12 = σ11*T5_12 + σ12*T5_22 + σ13*T5_23 + T5_11*σ12 + T5_12*σ22 + T5_13*σ23
    T8_13 = σ11*T5_13 + σ12*T5_23 + σ13*T5_33 + T5_11*σ13 + T5_12*σ23 + T5_13*σ33
    T8_23 = σ12*T5_13 + σ22*T5_23 + σ23*T5_33 + T5_12*σ13 + T5_22*σ23 + T5_23*σ33

    # T9 = σ⋅σ⋅γd⋅γd + γd⋅γd⋅σ⋅σ
    T9_11 = 2*(T4_11*T5_11 + T4_12*T5_12 + T4_13*T5_13)
    T9_22 = 2*(T4_12*T5_12 + T4_22*T5_22 + T4_23*T5_23)
    T9_33 = 2*(T4_13*T5_13 + T4_23*T5_23 + T4_33*T5_33)
    T9_12 = T4_11*T5_12 + T4_12*T5_22 + T4_13*T5_23 + T5_11*T4_12 + T5_12*T4_22 + T5_13*T4_23
    T9_13 = T4_11*T5_13 + T4_12*T5_23 + T4_13*T5_33 + T5_11*T4_13 + T5_12*T4_23 + T5_13*T4_33
    T9_23 = T4_12*T5_13 + T4_22*T5_23 + T4_23*T5_33 + T5_12*T4_13 + T5_22*T4_23 + T5_23*T4_33

    # Compute the integrity basis from scalar invariants
    # λ1 = tr(σ)
    λ1 = σ11 + σ22 + σ33

    # λ2 = tr(σ^2)
    λ2 = T4_11 + T4_22 + T4_33

    # λ3 = tr(γd^2)
    λ3 = T5_11 + T5_22 + T5_33

    # λ4 = tr(σ^3)
    λ4 = σ11*T4_11 + σ22*T4_22 + σ33*T4_33 + 2*(σ12*T4_12 + σ13*T4_13 + σ23*T4_23)

    # λ5 = tr(γd^3)
    λ5 = γd11*T5_11 + γd22*T5_22 + γd33*T5_33 + 2*(γd12*T5_12 + γd13*T5_13 + γd23*T5_23)

    # λ6 = tr(σ^2⋅γd^2)
    λ6 = T4_11*T5_11 + T4_22*T5_22 + T4_33*T5_33 + 2*(T4_12*T5_12 + T4_13*T5_13 + T4_23*T5_23)

    # λ7 = tr(σ^2⋅γd)
    λ7 = (T7_11 + T7_22 + T7_33)/2

    # λ8 = tr(σ⋅γd^2)
    λ8 = (T8_11 + T8_22 + T8_33)/2

    # λ9 = tr(σ⋅γd)
    λ9 = (T6_11 + T6_22 + T6_33) / 2f0

    # Run the integrity basis through a neural network
    model_inputs = [λ1;λ2;λ3;λ4;λ5;λ6;λ7;λ8;λ9]
    g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights) 
	# tst that this code is being executed. Plot should change. The code was indeed executing, 
	# and the solution did not change from its initial value. This must imply that the nonlinearity
	# has very little effect. 
    #g1,g2,g3,g4,g5,g6,g7,g8,g9 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
	
	# Save g1 through g9 per epoch. Once trained 
	#println("g1->g9:  $t, $g1, $g2, $g3, $g4, $g5, $g6, $g7, $g8, $g9")
	
    # Tensor combining layer
    F11 = g1 + g2*σ11 + g3*γd11 + g4*T4_11 + g5*T5_11 + g6*T6_11 + g7*T7_11 + g8*T8_11 + g9*T9_11
    F22 = g1 + g2*σ22 + g3*γd22 + g4*T4_22 + g5*T5_22 + g6*T6_22 + g7*T7_22 + g8*T8_22 + g9*T9_22
    F33 = g1 + g2*σ33 + g3*γd33 + g4*T4_33 + g5*T5_33 + g6*T6_33 + g7*T7_33 + g8*T8_33 + g9*T9_33
    F12 = g2*σ12 + g3*γd12 + g4*T4_12 + g5*T5_12 + g6*T6_12 + g7*T7_12 + g8*T8_12 + g9*T9_12
    F13 = g2*σ13 + g3*γd13 + g4*T4_13 + g5*T5_13 + g6*T6_13 + g7*T7_13 + g8*T8_13 + g9*T9_13
    F23 = g2*σ23 + g3*γd23 + g4*T4_23 + g5*T5_23 + g6*T6_23 + g7*T7_23 + g8*T8_23 + g9*T9_23
    
	if dct[:captureG]
		coef = [t, g1,g2,g3,g4,g5,g6,g7,g8,g9]
		push!(tdnn_coefs, coef)
		trace = [t, λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8, λ9]
		push!(tdnn_traces, trace)
		F = [t, F11, F22, F33, F12, F13, F23]
		push!(tdnn_Fs, F)
	end

    return F11,F22,F33,F12,F13,F23  # REINSTATE WHEN DEBUGGED
end

function dudt_univ!(du, u, p, t, gradv, dct)
	# the parameters are [NN parameters, ODE parameters)
	n_weights = dct[:n_weights]
    model_weights = p[1:n_weights]
    η0 = p[end - 1]
    τ = p[end]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t) 
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)

    # Run stress/strain through a TBNN
    γd = [γd11,γd22,γd33,γd12,γd13,γd23]
	F11,F22,F33,F12,F13,F23 = tbnn(u,γd,model_weights, dct[:model_univ], t)

    # The model differential equations
    dσ11 = η0*γd11/τ - σ11/τ + 2*v11(t)*σ11 + v21(t)*σ12 + v31(t)*σ13 + σ12*v21(t) + σ13*v31(t) - F11/τ
    dσ22 = η0*γd22/τ - σ22/τ + 2*v22(t)*σ22 + v12(t)*σ12 + v32(t)*σ23 + σ12*v12(t) + σ23*v32(t) - F22/τ
    dσ33 = η0*γd33/τ - σ33/τ + 2*v33(t)*σ33 + v13(t)*σ13 + v23(t)*σ23 + σ13*v13(t) + σ23*v23(t) - F33/τ
    dσ12 = η0*γd12/τ - σ12/τ + v11(t)*σ12 + v21(t)*σ22 + v31(t)*σ23 + σ11*v12(t) + σ12*v22(t) + σ13*v32(t) - F12/τ
    dσ13 = η0*γd13/τ - σ13/τ + v11(t)*σ13 + v21(t)*σ23 + v31(t)*σ33 + σ11*v13(t) + σ12*v23(t) + σ13*v33(t) - F13/τ
    dσ23 = η0*γd23/τ - σ23/τ + v12(t)*σ13 + v22(t)*σ23 + v32(t)*σ33 + σ12*v13(t) + σ22*v23(t) + σ23*v33(t) - F23/τ

    # Update in place
    du[1] = dσ11
    du[2] = dσ22
    du[3] = dσ33
    du[4] = dσ12
    du[5] = dσ13
    du[6] = dσ23
	nothing
end
