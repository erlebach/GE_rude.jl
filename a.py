diff --git a/Manifest.toml b/Manifest.toml
index a3db0d3..862a6c2 100644
--- a/Manifest.toml
+++ b/Manifest.toml
@@ -1387,10 +1387,10 @@ uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
 version = "1.4.1"
 
 [[deps.OrdinaryDiffEq]]
-deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "IfElse", "LinearAlgebra", "LinearSolve", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "NonlinearSolve", "Polyester", "PreallocationTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLNLSolve", "SimpleNonlinearSolve", "SnoopPrecompile", "SparseArrays", "SparseDiffTools", "StaticArrayInterface", "StaticArrays", "UnPack"]
-git-tree-sha1 = "a364df19a43c4a9520eeca693aa2e77b679a2b0c"
+deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "IfElse", "LinearAlgebra", "LinearSolve", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "NonlinearSolve", "Polyester", "PreallocationTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLNLSolve", "SimpleNonlinearSolve", "SnoopPrecompile", "SparseArrays", "SparseDiffTools", "StaticArrayInterface", "StaticArrays", "TruncatedStacktraces", "UnPack"]
+git-tree-sha1 = "5370a27bf89e6ac04517c6b9778295cdb7a411f8"
 uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
-version = "6.47.0"
+version = "6.48.0"
 
 [[deps.PCRE2_jll]]
 deps = ["Artifacts", "Libdl"]
@@ -1677,10 +1677,10 @@ uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
 version = "0.6.38"
 
 [[deps.SciMLBase]]
-deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
-git-tree-sha1 = "d3f3eaa16bdbee617c31c10324bc0b6a26ceaaac"
+deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SnoopPrecompile", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
+git-tree-sha1 = "fe55d9f9d73fec26f64881ba8d120607c22a54b0"
 uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
-version = "1.87.0"
+version = "1.88.0"
 
 [[deps.SciMLNLSolve]]
 deps = ["DiffEqBase", "LineSearches", "NLsolve", "Reexport", "SciMLBase"]
diff --git a/README b/README
index a333ad5..f555ade 100644
--- a/README
+++ b/README
@@ -48,3 +48,22 @@ Returned zeros at the end of tbnn
    - The warning appears. Why? 
 
 Moved the return [0.,,0.] statement to line 28 of rude_functions.jl after definition of T9_23
+----------------------------------------------------------------------
+2023-02-26_14:53
+The number of points is taken into account in the univ_loss function in rude_impl()
+	for k = range(1,trajectories,step=1)
+		σ12_pred = results[k][4,:]   # result density depends on :saveat
+		σ12_data = σ12_all[k]
+		loss += sum(abs2, σ12_pred - σ12_data)
+	end
+----------------------------------------------------------------------
+2023-02-26_15:00
+Strange phenomena. I am getting the same results independent of network complexity. The time to compute the 
+solution is the same independent of network complexity.  Yet, the yml file appears to have the correct metadata (in terms of the number of network weights for example). 
+Somehow, I am not executing the code I think I am). 
+
+In a whole series of runs, my loss is independent of the network parameters. That is SERIOUSLY INCORRECT!
+----------------------------------------------------------------------
+I am confused again. What is the relationship between Gisekus (model) and the equation we are seeking. In other 
+words, are we expecting agreement? What are the exact solutions to the equations?
+----------------------------------------------------------------------
diff --git a/myplots.jl b/myplots.jl
index b2fc35b..bcac75b 100644
--- a/myplots.jl
+++ b/myplots.jl
@@ -1,6 +1,6 @@
 
-function plot_solution(θ0, θi, tspan, σ0, p_giesekus; dct)
-    print("plot_solution: ", dct|>keys)
+function plot_solution(θ0, θi, p_system, tspan, σ0, p_giesekus; dct)
+    println("plot_solution: ", dct|>keys)
     # Define the simple shear deformation protocol
     v11(t) = 0
     v12(t) = 0
@@ -16,6 +16,7 @@ function plot_solution(θ0, θi, tspan, σ0, p_giesekus; dct)
 
     θ0 = [θ0; p_system]  # create single column vector
     θi = [θi; p_system]
+	@show θ0, θi
     ω = dct[:ω]
     γ = dct[:γ]
     
@@ -40,15 +41,17 @@ function plot_solution(θ0, θi, tspan, σ0, p_giesekus; dct)
         dudt!(du,u,p,t) = dudt_giesekus!(du,u,p,t,protocols[k])
         prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
 		sol_giesekus = solve(prob_giesekus,Tsit5(),saveat=dct[:saveat])
+
+        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k], dct)
     
         # Solve the UDE pre-training  (use parameters θ0)
-        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k], dct)
-        prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θ0) 
-		sol_ude_pre = solve(prob_univ, Tsit5(),abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])
+        prob_univ_pre = ODEProblem(dudt_ude!, σ0, tspan, θ0) 
+		sol_ude_pre = solve(prob_univ_pre, Tsit5(), abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])
         
-        # Solve the UDE post-training (use aprameters θi)
-        prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θi)
-		sol_ude_post = solve(prob_univ, abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])
+        # Solve the UDE post-training (use parameters θi)
+        prob_univ_post = ODEProblem(dudt_ude!, σ0, tspan, θi)
+		sol_ude_post = solve(prob_univ_post, abstol = 1e-7, reltol = 1e-6, saveat=dct[:saveat])
+
         plot_data!(plots, target[k], target_titles[k], sol_ude_pre, sol_ude_post, sol_giesekus)
     end
 
diff --git a/rude_functions.jl b/rude_functions.jl
index 0e7b65a..bef1362 100644
--- a/rude_functions.jl
+++ b/rude_functions.jl
@@ -2,16 +2,6 @@
 # Ideally, they should not access global variables (for portability). I am not sure that is the case. 
 
 function dudt_giesekus!(du, u, p, t, gradv)
-        # # Update in place
-        # du[1] = 0.
-        # du[2] = 0.
-        # du[3] = 0.
-        # du[4] = 0.
-        # du[5] = 0.
-        # du[6] = 0.
-        # return du
-
-
     # Destructure the parameters
     η0 = p[1]
     τ = p[2]
@@ -56,24 +46,9 @@ function dudt_giesekus!(du, u, p, t, gradv)
     du[5] = η0*γd13/τ - σ13/τ - (ω12*σ23 + ω13*σ33 - σ11*ω13 - σ12*ω23)/2 - F13/τ
     du[6] = η0*γd23/τ - σ23/τ - (ω23*σ33 - ω12*σ13 - σ12*ω13 - σ22*ω23)/2 - F23/τ
     ##
-
-    #= Update in place
-    du[1] = 0.
-    du[2] = 0.
-    du[3] = 0.
-    du[4] = 0.
-    du[5] = 0.
-    du[6] = 0.
-    return  # REMOVE THIS AND ABVOE du[i] = 0 ONCE DEBUGGED
-    =#
 end
 
-function tbnn(σ,γd,model_weights, model_univ)
-	# Remove next three lines once Debugging complete)
-    #model_inputs = [0.;0.;0.;0.;0.;0.;0.;0.;0.]
-    #g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights)  # Will this create the warning ?
-    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE. No warning generated. 
-
+function tbnn(σ, γd, model_weights, model_univ)
     # Tensor basis neural network (TBNN)
     # Unpack the inputs
     σ11,σ22,σ33,σ12,σ13,σ23 = σ
@@ -129,8 +104,6 @@ function tbnn(σ,γd,model_weights, model_univ)
     T9_13 = T4_11*T5_13 + T4_12*T5_23 + T4_13*T5_33 + T5_11*T4_13 + T5_12*T4_23 + T5_13*T4_33
     T9_23 = T4_12*T5_13 + T4_22*T5_23 + T4_23*T5_33 + T5_12*T4_13 + T5_22*T4_23 + T5_23*T4_33
 
-    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE
-
     # Compute the integrity basis from scalar invariants
     # λ1 = tr(σ)
     λ1 = σ11 + σ22 + σ33
@@ -157,14 +130,15 @@ function tbnn(σ,γd,model_weights, model_univ)
     λ8 = (T8_11 + T8_22 + T8_33)/2
 
     # λ9 = tr(σ⋅γd)
-    λ9 = (T6_11 + T6_22 + T6_33)/2
-
-    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE
+    λ9 = (T6_11 + T6_22 + T6_33) / 2f0
 
     # Run the integrity basis through a neural network
     model_inputs = [λ1;λ2;λ3;λ4;λ5;λ6;λ7;λ8;λ9]
-    g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights)  # model_univ not found
-    #return 0., 0., 0., 0., 0., 0.   # Generates warning. 
+    g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights) 
+	# tst that this code is being executed. Plot should change. The code was indeed executing, 
+	# and the solution did not change from its initial value. This must imply that the nonlinearity
+	# has very little effect. 
+    #g1,g2,g3,g4,g5,g6,g7,g8,g9 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
     
     # Tensor combining layer
     F11 = g1 + g2*σ11 + g3*γd11 + g4*T4_11 + g5*T5_11 + g6*T6_11 + g7*T7_11 + g8*T8_11 + g9*T9_11
@@ -174,16 +148,11 @@ function tbnn(σ,γd,model_weights, model_univ)
     F13 = g2*σ13 + g3*γd13 + g4*T4_13 + g5*T5_13 + g6*T6_13 + g7*T7_13 + g8*T8_13 + g9*T9_13
     F23 = g2*σ23 + g3*γd23 + g4*T4_23 + g5*T5_23 + g6*T6_23 + g7*T7_23 + g8*T8_23 + g9*T9_23
 
-    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE
     return F11,F22,F33,F12,F13,F23  # REINSTATE WHEN DEBUGGED
 end
 
 function dudt_univ!(du, u, p, t, gradv, dct)
-    # Destructure the parameters
-    # println("*** INSIDE dudt_univ! ***")
-    # println("typeof(dct): ", typeof(dct))
-    # println("dct: ", dct)
-    # println("dct keys: ", keys(dct) |> collect)
+	# the parameters are [NN parameters, ODE parameters)
 	n_weights = dct[:n_weights]
     model_weights = p[1:n_weights]
     η0 = p[end - 1]
@@ -222,27 +191,16 @@ function dudt_univ!(du, u, p, t, gradv, dct)
     du[4] = dσ12
     du[5] = dσ13
     du[6] = dσ23
-
-	#=
-    # Update in place
-    du[1] = 0.
-    du[2] = 0.
-    du[3] = 0.
-    du[4] = 0.
-    du[5] = 0.
-    du[6] = 0.
-    return  # REMOVE THIS AND ABVOE du[i] = 0 ONCE DEBUGGED
-	=#
 end
 
-function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories, dct)
+function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, dct)
     # Define the (default) ODEProblem
 	dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1], dct)
     prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)
 
     # Remake the problem for different protocols
     function prob_func(prob, i, repeat, dct)
-			dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[i], dct)
+		dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[i], dct)
         remake(prob, f=dudt_remade!, tspan=tspans[i])
     end
 
@@ -253,15 +211,17 @@ function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories, dct)
     sol = solve(prob, tsitdas_alg)
     =#
 
+	# EnsembleProblem: https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
     prob_func1(prob, i, repeat) = prob_func(prob, i, repeat, dct)
     ensemble_prob = EnsembleProblem(prob, prob_func=prob_func1)
 	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=dct[:saveat])  # original
-    #sim = solve(ensemble_prob, Rodas5(), ensemble, trajectories=trajectories, saveat=0.2)  # for stiff equation
+	# sim is returned by default
 end
 
 function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, dct)
     loss = 0
-    results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories, dct)
+	println("===> loss_univ: trajectories: $trajectories")
+    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, dct)
     for k = range(1,trajectories,step=1)
         σ12_pred = results[k][4,:]
         σ12_data = σ12_all[k]
@@ -294,7 +254,7 @@ function plot_data!(plots, targetk, target_titlek, sol_ude_pre, sol_ude_post, so
     halt = false
     if maximum(σ12_ude_post) > 10 || maximum(N1_ude_post) > 10 || maximum(N2_ude_post) > 10
         halt = true
-        print("halt: true")
+        println("halt: true")
     end
 
     # Plot the data
@@ -303,25 +263,34 @@ function plot_data!(plots, targetk, target_titlek, sol_ude_pre, sol_ude_post, so
     # When the blue line is not there, it is because the red and blue lines are superimposed. This means that the converged
     # state is near the initial state. This might happen where you run only a few epochs of the neural network. 
 
+	println("nb time points: ", sol_ude_post.t |> length)
+	println("sol_giesekus time points: ", sol_giesekus.t |> length)
+
+	mss = 1.25
+
     if targetk == "σ12"
-        plot!(sol_ude_post.t,σ12_ude_post, c=:red, lw=1.5, label="UDE-post") 
-        plot!(sol_ude_pre.t,σ12_ude_pre, c=:blue, ls=:dash, lw=3, title="σ12, v21=$target_titlek", label="UDE-pre") 
-        plot_σ12 = scatter!(sol_giesekus.t[1:2:end],σ12_data[1:2:end], c=:black, m=:o, ms=2, label="Giesekus") 
+        plot!(sol_ude_post.t, σ12_ude_post, c=:red, lw=1.5, label="UDE-post") 
+        plot!(sol_ude_pre.t, σ12_ude_pre, c=:blue, ls=:dash, lw=3, title="σ12, v21=$target_titlek", label="UDE-pre") 
+        scatter!(sol_giesekus.t[1:2:end], σ12_data[1:2:end], c=:black, m=:o, ms=mss, label="Giesekus") 
+		plot_σ12 = plot!(sol_giesekus.t[1:2:end], σ12_data[1:2:end], lw=0.5, c=:black)
         push!(plots, plot_σ12)
     elseif targetk == "N1"
-        plot!(sol_ude_post.t,N1_ude_post,c=:red, lw=1.5, label="UDE-post") 
-        plot!(sol_ude_pre.t,N1_ude_pre, c=:blue, ls=:dash, lw=3, title="N1, v21=$target_titlek", label="UDE-pre")  
-        plot_N1 = scatter!(sol_giesekus.t[1:2:end],N1_data[1:2:end], m=:o, ms=2, c=:black, label="Giesekus")
+        plot!(sol_ude_post.t, N1_ude_post, c=:red, lw=1.5, label="UDE-post") 
+        plot!(sol_ude_pre.t, N1_ude_pre, c=:blue, ls=:dash, lw=3, title="N1, v21=$target_titlek", label="UDE-pre")  
+        scatter!(sol_giesekus.t[1:2:end], N1_data[1:2:end], m=:o, ms=mss, c=:black, label="Giesekus")
+		plot_N1 = plot!(sol_giesekus.t[1:2:end], N1_data[1:2:end], lw=0.5, c=:black)
         push!(plots, plot_N1)
     elseif targetk == "N2"
-        plot!(sol_ude_post.t,N2_ude_post,c=:red, lw=1.5, label="UDE-post") 
-        plot!(sol_ude_pre.t,N2_ude_pre,c=:blue, lw=3, title="N2, v21=$target_titlek", ls=:dash, label="UDE-pre")
-        plot_N2 = scatter!(sol_giesekus.t[1:2:end], N2_data[1:2:end], m=:o, ms=2, c=:black, label="Giesekus") 
+        plot!(sol_ude_post.t, N2_ude_post, c=:red, lw=1.5, label="UDE-post") 
+        plot!(sol_ude_pre.t, N2_ude_pre, c=:blue, lw=3, title="N2, v21=$target_titlek", ls=:dash, label="UDE-pre")
+		scatter!(sol_giesekus.t[1:2:end], N2_data[1:2:end], m=:o, ms=mss, c=:black, label="Giesekus")
+		plot_N2 = plot!(sol_giesekus.t[1:2:end], N2_data[1:2:end], lw=0.5, c=:black) 
         push!(plots, plot_N2)
     elseif targetk == "ηE"
-        plot!(sol_ude_post.t,-N2_ude_post-N1_ude_post, lw=1.5, c=:red, label="UDE-post") 
-        plot!(sol_ude_pre.t,-N2_ude_pre-N1_ude_pre, lw=3, c=:blue, title="ηE=-N1-N2, v21=$target_titlek", ls=:dash, label="UDE-pre") 
-        plot_N2N1 = scatter!(sol_giesekus.t,-N2_data-N1_data, c=:black, m=:o, ms=2, label="Giesekus") 
+        plot!(sol_ude_post.t, -N2_ude_post-N1_ude_post, lw=1.5, c=:red, label="UDE-post") 
+        plot!(sol_ude_pre.t, -N2_ude_pre-N1_ude_pre, lw=3, c=:blue, title="ηE=-N1-N2, v21=$target_titlek", ls=:dash, label="UDE-pre") 
+        scatter!(sol_giesekus.t, -N2_data-N1_data, c=:black, m=:o, ms=mss, label="Giesekus") 
+		plot_N2N1 = plot!(sol_giesekus.t, -N2_data-N1_data, lw=0.5, c=:black)
         push!(plots, plot_N2N1)
     end  
     return plots, halt
diff --git a/rude_impl.jl b/rude_impl.jl
index 419e0bc..168f6cc 100644
--- a/rude_impl.jl
+++ b/rude_impl.jl
@@ -37,7 +37,7 @@ function plot_callback(θ, l, θ0, σ0, p_giesekus, step; dct)
 	dct[:nb_iter_optim] = nb_iter
     
     if nb_iter % step == 0
-        plots, halt = plot_solution(θ0, θ, tspan, σ0, p_giesekus; dct)
+        plots, halt = plot_solution(θ0, θ, p_giesekus, tspan, σ0, p_giesekus; dct)
         callback_plot = plot(plots..., plot_title="iteration $nb_iter, $(now())") 
 		display(callback_plot)
         fig_file_name = "callback_plot_$nb_iter.pdf"
@@ -49,7 +49,7 @@ end
 
 function single_run(dct)
 
-    # Define the simple shear deformation protocol
+	# Define the simple shear deformation protocol (all components except v21(t))
     v11(t) = 0
     v12(t) = 0
     v13(t) = 0
@@ -77,14 +77,14 @@ function single_run(dct)
 
     # Modify the choice of protcol from `rude-script.jl`
 
-    v21_1(t) = γs[1]*γ*cos(ωs[1]*ω*t)
-    v21_2(t) = γs[2]*γ*cos(ωs[2]*ω*t)
-    v21_3(t) = γs[3]*γ*cos(ωs[3]*ω*t)
-    v21_4(t) = γs[4]*γ*cos(ωs[4]*ω*t)
-    v21_5(t) = γs[5]*γ*cos(ωs[5]*ω*t)
-    v21_6(t) = γs[6]*γ*cos(ωs[6]*ω*t)
-    v21_7(t) = γs[7]*γ*cos(ωs[7]*ω*t)
-    v21_8(t) = γs[8]*γ*cos(ωs[8]*ω*t)
+    v21_1(t) = γs[1] * γ * cos(ωs[1] *ω * t)
+    v21_2(t) = γs[2] * γ * cos(ωs[2] *ω * t)
+    v21_3(t) = γs[3] * γ * cos(ωs[3] *ω * t)
+    v21_4(t) = γs[4] * γ * cos(ωs[4] *ω * t)
+    v21_5(t) = γs[5] * γ * cos(ωs[5] *ω * t)
+    v21_6(t) = γs[6] * γ * cos(ωs[6] *ω * t)
+    v21_7(t) = γs[7] * γ * cos(ωs[7] *ω * t)
+    v21_8(t) = γs[8] * γ * cos(ωs[8] *ω * t)
 
     # Generate one function for each protocol
     v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]
@@ -111,9 +111,6 @@ function single_run(dct)
     η0 = gie[:η0]
     τ = gie[:τ]
     α = gie[:α]
-    #η0 = 1
-    #τ = 1
-    #α = 0.8
     p_giesekus = [η0, τ, α]
     σ12_all = Any[]
     t_all = Any[]
@@ -130,11 +127,16 @@ function single_run(dct)
     dNN = dct[:dct_NN]
     act = dNN[:act]
     hid = dNN[:hid_layer]
+	println("nb points in hidden layer: ", hid)
+	println("rude.impl, now: ", now())
+
+	#----------------------- MODEL ---------
     model_univ = FastChain(FastDense(dNN[:in_layer], hid, act),
                         FastDense(hid, hid, act),
                         #FastDense(hid, hid, act),  # a second hidden layer
                         FastDense(hid, dNN[:out_layer]))
 	dct[:model_univ] = model_univ
+	#--------------------------------------
     
     # The protocol at which we'll start continuation training
     # (choose start_at > length(protocols) to skip training)
@@ -145,20 +147,27 @@ function single_run(dct)
     
     if start_at > 1
         # Load the pre-trained model if not starting from scratch
-        @load "tbnn.bson" θi
+		println("Load a pre-trained model")
+		println("Loading a pre-trained network is temporarily disabled")
+        #@load "tbnn.bson" θi
         p_model = θi
         n_weights = length(θi)
-		dct[:n_weights] = n_weights
     else
         # The model weights are destructured into a vector of parameters
+		println("Train the model from scratch")
         p_model = initial_params(model_univ)
         n_weights = length(p_model)
         p_model = zeros(n_weights)
-		dct[:n_weights] = n_weights
     end
+
+	println("n_weights: ", n_weights)
+	dct[:n_weights] = n_weights
     
     # Parameters of the linear response (η0,τ)
-    p_system = Float32[1, 1]
+    gie = dct[:dct_giesekus]
+    η0 = gie[:η0]
+    τ = gie[:τ]
+	p_system = Float32[η0, τ]  # change 2023-02-26_15:40
     
     θ0 = zeros(size(p_model))
     θi = p_model
@@ -189,17 +198,19 @@ function single_run(dct)
 		# Original code
         #parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_plot, sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=dct[:maxiters]) 
 		# default lr is 1.e-3
-        parameter_res = Optimization.solve(optprob, Optimisers.Adam(1f-3), callback=cb_plot, sensealg = ReverseDiffVJP(true), allow_f_increases=true, maxiters=dct[:maxiters])
+        parameter_res = Optimization.solve(optprob, Optimisers.Adam(1f-3), 
+										   callback=cb_plot, sensealg = ReverseDiffVJP(true), 
+										   allow_f_increases=true, maxiters=dct[:maxiters])
+		# final network parameters
         θi = parameter_res.u
-        push!(out_files, "tbnn_k=" * string(k))
-        #@save "tbnn.bson" θi
-        @save out_files[end] θi
+        #push!(out_files, "tbnn_k=" * string(k))
+        #@save out_files[end] θi
     end
     
     # Build full parameter vectors for model testing
-    θ0 = [θ0; p_system]  # create single column vector
-    θi = [θi; p_system]
-    plots, halt = plot_solution(θ0, θi, tspan, σ0, p_giesekus; dct)
+    #θ0 = [θ0; p_system]  # create single column vector
+    #θi = [θi; p_system]
+    plots, halt = plot_solution(θ0, θi, p_system, tspan, σ0, p_giesekus; dct)
 	final_plot = plot(plots..., plot_title="Last training step, $(now())")
     display(final_plot)
     return final_plot 
diff --git a/rude_script.jl b/rude_script.jl
index 2ef86f6..43d0c13 100644
--- a/rude_script.jl
+++ b/rude_script.jl
@@ -12,22 +12,22 @@ using BSON: @save, @load
 using Dates
 
 # Need a global function
-include("rude_functions.jl")
-include("myplots.jl")
-include("rude_impl.jl")
+include("./rude_functions.jl")
+include("./myplots.jl")
+include("./rude_impl.jl")
 
 # There are 8 protocols for each value of ω0 and γ0. So if you have 4 pairs (ω0, γ0), 
 # you will have 8 protocols for each pair. (That is not what Sachin's problem proposes). 
 # So perhaps one has to generalize how the protocols are stored in the main program to run 
 # a wider variety of tests. Ask questions, Alex. 
 
-print("after includes: ", now())
+println("after includes: ", now())
 
 function setup_dict_production()
     dct_params = Dict()
     dct_params[:ω0] = [100.f0]
-    dct_params[:ω0] = [1f0]
-    dct_params[:γ0] = [.1f0] # not used
+    dct_params[:ω0] = [5f0]
+    dct_params[:γ0] = [1f0] # not used
 
 
     # Create a Dictionary with all parameters
@@ -38,7 +38,7 @@ function setup_dict_production()
     dct[:Ncycles] = 3
     dct[:nb_pts_per_cycle] = 80  # Perhaps increase this will increase accuracy? 
     dct[:nb_protocols] = 8  # set to 1 to run faster. Set to 8 for more accurate results
-    dct[:skip_factor] = 100
+    dct[:skip_factor] = 100  # callback every skip_factor
     dct[:dct_giesekus] = Dict()
     # Next two lines are not used yet
     dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
@@ -53,8 +53,10 @@ function setup_dict_production()
     dNN[:nb_hid_layers] = 1
     dNN[:in_layer] = 9
     dNN[:out_layer] = 9
-    dNN[:hid_layer] = 64 # nb points in th elayer
+	dNN[:hid_layer] = 32 # (TOO LOW) # 64 # nb points in th elayer
     dNN[:act] = tanh
+    #dNN[:act] = relu
+    #dNN[:act] = identity
 	dct[:losses] = []    # Accumulate losses
     return dct, dct_params
 end
@@ -65,7 +67,7 @@ function setup_dict_testing()
     dct_params[:γ0] = [.1f0] # not used
     dct[:maxiters] = 200
     dct[:nb_protocols] = 1
-    dct[:skip_factor] = 10
+    dct[:skip_factor] = 1
     dNN = dct[:dct_NN]
     dNN[:nb_hid_layers] = 1
     dNN[:in_layer] = 9
@@ -78,6 +80,8 @@ end
 # ============== END DICTIONARY DEFINITIONS ==================
 production = true  # set to false when testing the code, for faster calculations
 
+# ============== START of SCRIPT PROPER ==================
+
 if production == true
     # Production: parameters set to generate converged solutions
     dct, dct_params = setup_dict_production()
@@ -112,8 +116,8 @@ for o in dct_params[:ω0]
 		dct[:T] = (2. * π / dct[:ω]) * dct[:Ncycles]  |> nearest_multiple_of_10 |> Float32
 		dct[:saveat] = dct[:T] / dct[:nb_pts_per_cycle] / dct[:Ncycles] |> Float32 # where to save the solution
 
-		dct[:T] = 20.f0  #
-		dct[:saveat] = 0.2 #  (60 points over a timespan of 12)
+		dct[:T] = 12.f0  #
+		dct[:saveat] = 0.02 #  (1200 points over a timespan of 12)
 
 		# Trick to make sure that saveat is always a submultiple of the time span. Not used. 
         # dct[:saveat] = (saveat=saveat[2:end-1], save_start=true, save_end=true)
@@ -123,7 +127,7 @@ for o in dct_params[:ω0]
 		println("saveat time: ", dct[:saveat])
 
 		YAML.write_file("latest_dict_run$run.yml", dct)
-		print("before single_run, now: ", now())
+		println("before single_run, now: ", now())
         figure = single_run(dct)
         # deepcopy will make sure that the results is different than dct
         # Without it, the dictionaries saved in the list will all be the same
