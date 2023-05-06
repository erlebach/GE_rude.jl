# How to transform [[1,2,3], [4,5,6]] into a 2D array
using NPZ
#
a = []
push!(a, [1,2,3])
push!(a, [7,2,3])
push!(a, [10,2,3])
push!(a, [19,23,3])
println(size(hcat(a)))
println(size(vcat(a)))
println(typeof(a))

println(reduce(hcat, a) |> size) # (3,4). Each set is a column. 
NPZ.npzwrite("tdnn.npz", reduce(hcat, a))
println(reduce(hcat, a) |> transpose |> size) # (4,3). Each set is a row. 
