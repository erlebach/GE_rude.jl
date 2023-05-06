using Plots
using BSON: @load
using NPZ

@load "losses.bson" losses
@load "targets.bson" targets
tdnn_coefs = npzread("tdnn_coefs.npz")
tdnn_traces = npzread("tdnn_traces.npz")
tdnn_Fs = npzread("tdnn_Fs.npz")

println(tdnn_coefs |> size)
println(tdnn_traces |> size)
println(tdnn_Fs |> size)

t = tdnn_coefs[1,:]
coefs = Dict()
plots = []
for i in 1:9
	coefs[i] = tdnn_coefs[i+1,:]
	push!(plots, plot(t, coefs[i]))
end

gplot = plot(plots..., layout=(3,3)) 
savefig(gplot, "gplot.pdf")

