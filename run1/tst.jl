function myadd(a,b)
	return a+b
end

println(myadd(3,4))

@code_warntype myadd(3,4)
