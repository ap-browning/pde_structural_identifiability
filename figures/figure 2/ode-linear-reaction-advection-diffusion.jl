using DifferentialEquations
using DiffEqOperators
using Plots, StatsPlots, ColorSchemes
using LinearAlgebra
using Interpolations
using Distributions
using AdaptiveMCMC
using .Threads

# Set parameter values
p₁ = p₂ = p₃ = p₄ = p₅ = p₆ = 0.1; p₂ = 0.2
v₀ = 1.0

# Parameter vector
p = [p₁,p₂,p₃,p₄,p₅,p₆,v₀]

# Setup temporal and spatial domain
tmax = 10.0

# Initial condition (known)
n₀ = 3.0

# PDE solve
function solve_model(p)

    # Obtain parameters
    p₁,p₂,p₃,p₄,p₅,p₆,v₀ = p
    
    # Initial conditions
    u₀ = n₀ - v₀

    # ODE
    function ode!(dx,x,p,t)
        u,v = x
        dx[1] = p₁ * u + p₂ * v .+ p₃
        dx[2] = p₄ * u + p₅ * v .+ p₆
    end

    solve(ODEProblem(ode!,[u₀;v₀],(0.0,tmax)),RK4())

end

## Figure 1
sol = solve_model(p);
pltode = plot(t -> sum(sol(t)),xlim=(0.0,tmax),c=:black,lw=2.0,label="n(t)")
plot!(pltode,t -> sol(t)[2],c=palette(:Blues_7,rev=true)[3],lw=2.0,label="v(t)")

# Compensate parameters
β₁ = p₁ * p₅ - p₂ * p₄
β₂ = p₁ + p₅ 
β₃ = p₃ * (p₅ - p₄) + p₆ * (p₁ - p₂)
β₄ = p₃ + p₆ + (p₁ + p₄) * n₀ + (-p₁ + p₂ - p₄ + p₅) * v₀
β = [β₁,β₂,β₃,β₄]

p̂₁ = p₁
p̂₂ = 2 * p₂
p̂₃ = p₃
p̂₄ = p₄ / 2
p̂₅ = p₅
p̂₆ = (β₃ + p̂₃*p̂₄ - p̂₃*p̂₅) / (p̂₁ - p̂₂)
v̂₀ = (p̂₃ + p̂₆ + (p̂₁ + p̂₄)*n₀ - β₄) / (p̂₁ - p̂₂ + p̂₄ - p̂₅)

β̂₁ = p̂₁ * p̂₅ - p̂₂ * p̂₄
β̂₂ = p̂₁ + p̂₅ 
β̂₃ = p̂₃ * (p̂₅ - p̂₄) + p̂₆ * (p̂₁ - p̂₂)
β̂₄ = p̂₃ + p̂₆ + (p̂₁ + p̂₄) * n₀ + (-p̂₁ + p̂₂ - p̂₄ + p̂₅) * v̂₀
β̂ = [β̂₁,β̂₂,β̂₃,β̂₄]

[β β̂]

p̂ = [p̂₁,p̂₂,p̂₃,p̂₄,p̂₅,p̂₆,v̂₀]

# Plot curves with compensate parameters
sol = solve_model(p̂);
plot!(pltode,t -> sum(sol(t)),xlim=(0.0,tmax),c=:red,ls=:dash,lw=2.0,label="n, p₂ → 2 * p₂")
plot!(pltode,t -> sol(t)[2],xlim=(0.0,tmax),c=:orange,ls=:dash,lw=2.0,label="v, p₂ → 2 * p₂")