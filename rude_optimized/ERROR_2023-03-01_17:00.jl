ERROR: LoadError: MethodError: no method matching Vector{Float32}(::StaticArraysCore.SMatrix{3, 3, Float64, 9})
Stacktrace:
 [1] convert at ./array.jl:617
 [2] setproperty! at ./Base.jl:39
 [3] initialize! at /Users/erlebach/.julia/packages/OrdinaryDiffEq/W3SVv/src/perform_step/low_order_rk_perform_step.jl:700
 [4] #__init#627 at /Users/erlebach/.julia/packages/OrdinaryDiffEq/W3SVv/src/solve.jl:499
 [5] #__solve#626 at /Users/erlebach/.julia/packages/OrdinaryDiffEq/W3SVv/src/solve.jl:5
 [6] #solve_call#22 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:509
 [7] #solve_up#29 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:932
 [8] #solve#27 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:842
 [9] #_concrete_solve_adjoint#298 at /Users/erlebach/.julia/packages/SciMLSensitivity/bEthl/src/concrete_solve.jl:629
 [10] #_concrete_solve_adjoint#260 at /Users/erlebach/.julia/packages/SciMLSensitivity/bEthl/src/concrete_solve.jl:163
 [11] #_solve_adjoint#53 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:1348
 [12] #rrule#51 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:1301
 [13] chain_rrule_kw at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/chainrules.jl:235
 [14] macro expansion at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [15] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:9
 [16] _apply at ./boot.jl:816
 [17] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [18] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [19] _pullback at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:842
 [20] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [21] _apply at ./boot.jl:816
 [22] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [23] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [24] _pullback at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:832
 [25] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [26] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:92
 [27] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [28] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:87
 [29] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:146
 [30] #8 at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/ext/DiffEqBaseZygoteExt.jl:24
 [31] responsible_map at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:139
 [32] ∇responsible_map at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/ext/DiffEqBaseZygoteExt.jl:24
 [33] adjoint at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/ext/DiffEqBaseZygoteExt.jl:51
 [34] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [35] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:145
 [36] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [37] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:144
 [38] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [39] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:155
 [40] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [41] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:151
 [42] macro expansion at ./timing.jl:382
 [43] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:56
 [44] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [45] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/ensemble/basic_ensemble_solve.jl:45
 [46] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [47] _apply at ./boot.jl:816
 [48] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [49] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [50] _pullback at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:956
 [51] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [52] _apply at ./boot.jl:816
 [53] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [54] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [55] _pullback at /Users/erlebach/.julia/packages/DiffEqBase/JH4gt/src/solve.jl:952
 [56] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [57] _pullback at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_functions.jl:257
 [58] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [59] _pullback at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_functions.jl:264
 [60] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [61] _pullback at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_impl.jl:182
 [62] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [63] _pullback at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_impl.jl:193
 [64] _apply at ./boot.jl:816
 [65] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [66] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [67] _pullback at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/scimlfunctions.jl:3904
 [68] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [69] _apply at ./boot.jl:816
 [70] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [71] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [72] _pullback at /Users/erlebach/.julia/packages/Optimization/XjqVZ/src/function/zygote.jl:30
 [73] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [74] _apply at ./boot.jl:816
 [75] adjoint at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/lib/lib.jl:203
 [76] _pullback at /Users/erlebach/.julia/packages/ZygoteRules/AIbCs/src/adjoint.jl:65
 [77] _pullback at /Users/erlebach/.julia/packages/Optimization/XjqVZ/src/function/zygote.jl:34
 [78] _pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface2.jl:0
 [79] pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface.jl:44
 [80] pullback at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface.jl:42
 [81] gradient at /Users/erlebach/.julia/packages/Zygote/g2w9o/src/compiler/interface.jl:96
 [82] #157 at /Users/erlebach/.julia/packages/Optimization/XjqVZ/src/function/zygote.jl:32
 [83] macro expansion at /Users/erlebach/.julia/packages/OptimizationOptimisers/KGKWE/src/OptimizationOptimisers.jl:36
 [84] macro expansion at /Users/erlebach/.julia/packages/Optimization/XjqVZ/src/utils.jl:37
 [85] #__solve#1 at /Users/erlebach/.julia/packages/OptimizationOptimisers/KGKWE/src/OptimizationOptimisers.jl:35
 [86] #solve#552 at /Users/erlebach/.julia/packages/SciMLBase/gTrkJ/src/solve.jl:85
 [87] single_run at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_impl.jl:199
 [88] top-level scope at /Users/erlebach/src/2022/rude/giesekus/GE_rude.jl/rude_optimized/rude_script.jl:162
