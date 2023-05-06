using DifferentialEquations

function solve3x3!(dσ, σ, p, t)
    dσ .= -σ
end

tspan = (0., 2.)
p_simple = [2.]
σ0 = [1. 0. 0. ; 0. 2. 0. ; 3. 0. 0.]
dσ = zeros(3,3)
prob = ODEProblem(solve3x3!, σ0, tspan, p_simple)
solve_simple = solve(prob, Tsit5(), saveat=1.0);  # Do not show output of solve()
solve_simple.u