# GE_rude.jl
Rude experiments with Julia

Starting from the Rude implementation at https://github.com/krlennon/rude.git, we have refactored the code to allow multiple simulations to run consecutaively without paying the penalty of Julia start. Dictionaries of parameters are stored to Yaml files for further investigation. Plotting is performed with the Plots.jl library.
