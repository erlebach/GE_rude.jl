using Plots
using Glob
using CSV
using DataFrames 

#F_files = PTT_F*
F_files = glob("PTT_F_*txt")
σ_files = glob("PTT_σ_*txt")
g_files = glob("PTT_g_*txt")
λ_files = glob("PTT_lambda_*txt")
println(λ_files)

# Preprocess all files
all_files = vcat(F_files, σ_files, λ_files, g_files)

#----------------------------------------------------------------------
# Identify the first line that starts with 12.0
# ChatGPT:  I have file with numbers. I'd like to identify the first line that starts with 12.0, skip the next line, and
# output the remaining lines to a new file.
function get_data(filename)
    # Read in file
    file = open(filename)
    lines = readlines(file)
    close(file)

    # Find the first line that starts with 12.0
	start = 0
    for i in 1:length(lines)
        if occursin("12.0", lines[i])
            start = i
            break
        end
    end

    # Skip the next line
    stop = start + 2

    # Output the remaining lines to a new file
    file = open(filename)
    lines = readlines(file)
    close(file)
    lines = lines[stop:end]
	filename = filename*"_"
    file = open(filename, "w")
    for i in 1:length(lines)
        write(file, lines[i])
        write(file, "\n")
    end
    close(file)
end
#----------------------------------------------------------------------

for f in all_files
	get_data(f)
end

F_files = glob("PTT_F_*.txt_")
σ_files = glob("PTT_σ_*.txt_")
g_files = glob("PTT_g_*.txt_")
λ_files = glob("PTT_lambda_*.txt_")

# Plot F data the data in files ending in _
function plot_files(files, variable)
    plts = []
    fontsz = 4
    legcol = RGBA(1.,1.,1.,.6)

    function set_params()
        if variable == "F"
            headers = [:F11, :F22, :F33, :F12, :F13, :F23]
            labels = ["F11" "F22" "F33" "F12"]
        elseif variable == "g"
            headers = [:g1, :g2, :g3, :g4, :g5, :g6, :g7, :g8, :g9]
            labels = ["g1" "g2" "g3" "g4" "g5" "g6" "g7" "g8" "g9"]
        elseif variable == "λ"
            headers = [:λ1, :λ2, :λ3, :λ4, :λ5, :λ6, :λ7, :λ8, :λ9] 
            labels = ["λ1" "λ2" "λ3" "λ4" "λ5" "λ6" "λ7" "λ8" "λ9"]
        elseif variable == "σ"
            headers = [:σ11, :σ22, :σ33, :σ12, :σ13, :σ23]
            labels = ["σ11" "σ22" "σ33" "σ12"]
        else
            println("Invalid variable")
            return
        end
        return headers, labels
    end

    headers, labels = set_params()

    for f in files
        # Only process target 1
        # how do I negate occursin?
        if !occursin("target1", f)
            continue
        end
        # extend [:t] by [:F11, :F22, :F33, :F12, :F13, :F23]
	    data = CSV.read(f, DataFrame, delim=",", header=vcat([:t], headers))
        y = Matrix(data[:, 2:1+length(labels)])
        println(size(y))
        #plt = scatter(data[:,1], Matrix(data[:, 2:5]), xlabel="t", label=headers, title=f, ms=1,
        #guidefontsize=fontsz, tickfontsize=fontsz, legendfontsize=fontsz, titlefontsize=fontsz, 
        #background_color_legend=legcol, foreground_color_legend=legcol, mec=:none, msw=0)
        plt = plot(data[:,1], y, xlabel="t", labels=labels, title=f, ms=1,
        guidefontsize=fontsz, tickfontsize=fontsz, legendfontsize=fontsz, titlefontsize=fontsz, 
        background_color_legend=legcol, foreground_color_legend=legcol, mec=:none, msw=0)
        push!(plts, plt)
    end
    return plts
end

plt = plot(plot_files(F_files, "F")..., layout=(2,2));
savefig(plt, "F.pdf")
plt = plot(plot_files(σ_files, "σ")..., layout=(2,2));
savefig(plt, "σ.pdf")
plt = plot(plot_files(g_files, "g")..., layout=(2,2));
savefig(plt, "g.pdf")
plt = plot(plot_files(λ_files, "λ")..., layout=(2,2));
savefig(plt, "λ.pdf")

