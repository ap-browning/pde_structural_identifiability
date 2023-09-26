using DifferentialEquations
using DiffEqOperators
using Plots, StatsPlots, ColorSchemes
using LinearAlgebra
using Interpolations
using Distributions
using AdaptiveMCMC
using .Threads
using NLsolve

include("defaults.jl")

# Set parameter values
D₁ = 20.0; D₂ = 100.0
χ = 1.0
k = 1.0
α = 1.0

# Parameter vector
p = [D₁,D₂,χ,k,α]

# Initial conditions...
ρ₀ = x -> -(tanh(1.0*(x-20))+1)/2 .+ 1.0
c₀ = x -> 0.0 * x

# Setup temporal and spatial domain
tmax = 20.0
xmax = 100.0

# PDE solve
function solve_model(p,c₀;N=101)
    # Spatial domain
    x    = range(0.0,100.0,N)
    Δx   = x[2] - x[1]

    # Obtain parameters
    D₁,D₂,χ,k,α = p
    
    # PDE
    bc = Neumann0BC(Δx)
    ∂ₓ = UpwindDifference(1,1,Δx,N,1.0)
    ∂ₓₓ = CenteredDifference(2,2,Δx,N)

    function pde!(dy,y,p,t)
        ρ,c = y[1:N],y[N+1:end]
        dρ,dc = (@view dy[1:N]), (@view dy[N+1:end])
        dρ .= D₁ * ∂ₓₓ * bc * ρ - χ * (∂ₓₓ * bc * c) .* ρ - χ * (∂ₓ * bc * c) .* (∂ₓ * bc * ρ)
        dc .= D₂ * ∂ₓₓ * bc * c - k * c + α * ρ
    end

    solve(ODEProblem(pde!,[ρ₀.(x);c₀.(x)],(0.0,tmax))),x

end

## Figure 1
N = 501
sol,x = solve_model(p,c₀;N);
plt_ρ = [plot(palette=palette(:Greens_7,rev=true)[2:end],legend=:none,title="$t d") for t in 0:10:20]
plt_c = [plot(palette=palette(:Blues_7,rev=true)[2:end],legend=:none,title="$t d") for t in 0:10:20]
for (i,t) = enumerate(0.0:10.0:20.0)
    ρ,c = sol(t)[1:length(x)],sol(t)[length(x)+1:end]
    plot!(plt_ρ[i],x,ρ,lw=2.0)
    plot!(plt_c[i],x,c,lw=2.0)
end

# Compensate parameters
χ̄ = 2.0
ᾱ = 0.5
p̄ = [D₁,D₂,χ̄,k,ᾱ]

sol,x = solve_model(p̄,c₀;N);
for (i,t) = enumerate(0.0:10.0:20.0)
    ρ,c = sol(t)[1:length(x)],sol(t)[length(x)+1:end]
    plot!(plt_ρ[i],x,ρ,lw=2.0,c=:red,ls=:dash)
    plot!(plt_c[i],x,c,lw=2.0,c=:red,ls=:dash)
end

# Assemble figure
fig3 = plot(plt_ρ...,plt_c...,xlabel="t",ylabel="ρ",ylim=(0.0,1.0),widen=true)
for i = 4:6
    plot!(fig3,subplot=i,ylim=(0.0,0.7),ylabel="c")
end
fig3

savefig(fig3,"$(@__DIR__)/fig3.svg")