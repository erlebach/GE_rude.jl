function my()
a = true
if a
	global v = 3
else 
	global v = 4
end
println(v)
end

my()
