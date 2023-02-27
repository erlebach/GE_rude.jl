
function plot_solution(θ0, θi, p_system, tspan, σ0, p_giesekus; dct)
    println("plot_solution: ", dct|>keys)
    # Define the simple shear deformation protocol
    v11(t) = 0
    v12(t) = 0
    v13(t) = 0
    v22(t) = 0
    v23(t) = 0
    v31(t) = 0
    v32(t) = 0
    v33(t) = 0

    # Parameters of the linear response (η0,τ)
    p_system = Float32[1, 1]

    θ0 = [θ0; p_system]  # create single column vector
    θi = [θi; p_system]
	@show θ0, θi
    ω = dct[:ω]
    γ = dct[:γ]
    
    # Test the UDE on a new condition (different ω, different γ)
    v21_1(t) = 2*cos(3*ω*t/4)
    v21_2(t) = 2*cos(ω*t)
    v21_3(t) = 2*cos(ω*t)  # same as v21_2
    v21_4(t) = 1.5f0
    gradv_1 = [v11,v12,v13,v21_1,v22,v23,v31,v32,v33] 
    gradv_2 = [v11,v12,v13,v21_2,v22,v23,v31,v32,v33]
    gradv_3 = [v11,v12,v13,v21_3,v22,v23,v31,v32,v33]
    gradv_4 = [v11,v12,v13,v21_4,v22,v23,v31,v32,v33]
    protocols = [gradv_1, gradv_2, gradv_3, gradv_4]
    target = ["σ12","N1","N2","σ12"]
    target_titles = ["2cos(3ωt/4)", "2cos(ωt)", "2cos(ωt)", "1.5"]
	tspan = (0.0f0, dct[:T])
    
    plots = []
    halt = false
    for k = range(1,length(protocols),step=1)  # ORIGINAL
        # Solve the Giesekus model (σ11, σ22, σ33, σ12, σ13, σ23)
        dudt!(du,u,p,t) = dudt_giesekus!(du,u,p,t,protocols[k])
        prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
		sol_giesekus = solve(prob_giesekus,Tsit5(),saveat=dct[:saveat])

        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k], dct)
    
        # Solve the UDE pre-training  (use parameters θ0)
        prob_univ_pre = ODEProblem(dudt_ude!, σ0, tspan, θ0) 
		sol_ude_pre = solve(prob_univ_pre, Tsit5(), abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])
        
        # Solve the UDE post-training (use parameters θi)
        prob_univ_post = ODEProblem(dudt_ude!, σ0, tspan, θi)
		sol_ude_post = solve(prob_univ_post, abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])

        plot_data!(plots, target[k], target_titles[k], sol_ude_pre, sol_ude_post, sol_giesekus)
    end

    return plots, halt
end
