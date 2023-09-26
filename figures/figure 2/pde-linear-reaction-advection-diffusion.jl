using DifferentialEquations
using DiffEqOperators
using Plots, StatsPlots, ColorSchemes
using LinearAlgebra
using Interpolations
using Distributions
using AdaptiveMCMC
using .Threads
using NLsolve

include("../defaults.jl")

# Set parameter values
p₁ = p₂ = p₃ = p₄ = p₅ = p₆ = 0.1; p₂ = 0.2
D₁ = 20.0; D₂ = 10.0
α₁ = -5.0
α₂ = 5.0

# Parameter vector
p = [D₁,D₂,α₁,α₂,p₁,p₂,p₃,p₄,p₅,p₆]

# Initial conditions...
u₀unknown = x -> -(tanh(1.0*(x-20))+1)/2 .+ 2.0
v₀ = x -> (tanh(1.0*(x-40))+1)/2 .+ 1.0
n₀ = x -> u₀unknown(x) + v₀(x)

# Setup temporal and spatial domain
tmax = 10.0
xmax = 100.0

# PDE solve
function solve_model(p,v₀;N=101)
    # Spatial domain
    x    = range(0.0,100.0,N)
    Δx   = x[2] - x[1]

    # Obtain parameters
    D₁,D₂,α₁,α₂,p₁,p₂,p₃,p₄,p₅,p₆ = p
    
    # Initial conditions
    u₀ = x -> n₀(x) - v₀(x)

    # PDE
    bc = Neumann0BC(Δx)
    α₁_∂ₓ = UpwindDifference(1,1,Δx,N,α₁)
    α₂_∂ₓ = UpwindDifference(1,1,Δx,N,α₂)
    ∂ₓₓ = CenteredDifference(2,2,Δx,N)

    function pde!(dy,y,p,t)
        u,v = y[1:N],y[N+1:end]
        du,dv = (@view dy[1:N]), (@view dy[N+1:end])
        du .= D₁ * ∂ₓₓ * bc * u + α₁_∂ₓ * bc * u + p₁ * u + p₂ * v .+ p₃
        dv .= D₂ * ∂ₓₓ * bc * v + α₂_∂ₓ * bc * v + p₄ * u + p₅ * v .+ p₆
    end

    solve(ODEProblem(pde!,[u₀.(x);v₀.(x)],(0.0,tmax))),x

end

## Figure 1
N = 501
sol,x = solve_model(p,v₀;N);
pu₁ = plot(palette=palette(:Greens_7,rev=true),legend=:none)
pv₁ = plot(palette=palette(:Blues_7,rev=true),legend=:none)
pn = plot(palette=palette(:grayC,6,rev=true),legend=:none)
for t = 0.0:2.0:6.0
    u,v = sol(t)[1:length(x)],sol(t)[length(x)+1:end]
    w = u + v
    plot!(pu₁,x,u,lw=2.0)
    plot!(pv₁,x,v,lw=2.0)
    plot!(pn,x,w,lw=2.0)
end
plot(pu₁,pv₁,pn)

## Compensate parameters
β₁ = p₂ * p₄
β₂ = p₃ * (p₅ - p₄) + p₆ * (p₁ - p₂)
β = [β₁,β₂]

p̂₁ = p₁
p̂₂ = 2 * p₂
p̂₃ = p₃
p̂₄ = p₄ / 2
p̂₅ = p₅
p̂₆ = (β₃ + p̂₃*p̂₄ - p̂₃*p̂₅) / (p̂₁ - p̂₂)
p̂ = [D₁,D₂,α₁,α₂,p̂₁,p̂₂,p̂₃,p̂₄,p̂₅,p̂₆]

β̂₁ = p̂₂ * p̂₄
β̂₂ = p̂₃ * (p̂₅ - p̂₄) + p̂₆ * (p̂₁ - p̂₂)
β̂ = [β̂₁,β̂₂]
[β β̂]

## Find adjusted initial condition

###########################
## n₍₀,₁₎(x,0) with the original parameters
###########################

    # Spatial domain
    x    = range(0.0,100.0,N)
    Δx   = x[2] - x[1]

    # Initial conditions
    u₀ = x -> n₀(x) - v₀(x)
    u,v = u₀.(x),v₀.(x)

    # γ as a vector
    # bc = Neumann0BC(Δx)
    # α₁_∂ₓ = UpwindDifference(1,1,Δx,N,α₁)
    # α₂_∂ₓ = UpwindDifference(1,1,Δx,N,α₂)
    # ∂ₓₓ = CenteredDifference(2,2,Δx,N)
    # γ = D₁ * ∂ₓₓ * bc * u + α₁_∂ₓ * bc * u + p₁ * u + p₂ * v .+ p₃ + 
    #     D₂ * ∂ₓₓ * bc * v + α₂_∂ₓ * bc * v + p₄ * u + p₅ * v .+ p₆

    # γ as a function (smoother)
    using ForwardDiff
    u,v = u₀,v₀
    ∂ₓ = ForwardDiff.derivative
    ∂ₓₓ = (func,x) -> ∂ₓ(x -> ∂ₓ(func,x),x)
    γ = x -> D₁ * ∂ₓₓ(u,x) + α₁ * ∂ₓ(u,x) + p₁ * u(x) + p₂ * v(x) .+ p₃ + 
             D₂ * ∂ₓₓ(v,x) + α₂ * ∂ₓ(v,x) + p₄ * u(x) + p₅ * v(x) .+ p₆

    # LHS
    n = n₀
    f = x -> γ(x) - (p̂₃ + p̂₆ + (p̂₁ + p̂₄) * n(x) + α₁ * ∂ₓ(n,x) + D₁ * ∂ₓₓ(n,x))

    # RHS coefficients
    ω₁ = (-p̂₁ + p̂₂ - p̂₄ + p̂₅)
    ω₂ = α₂ - α₁
    ω₃ = D₂ - D₁

    bc = Neumann0BC(Δx)
    ω₂_Δₓ = UpwindDifference(1,1,Δx,N,ω₂)
    Δₓₓ = CenteredDifference(2,2,Δx,N)

    # Function of which to find the root
    func(v̂) = ω₁ * v̂ + ω₂_Δₓ * bc * v̂ + ω₃ * Δₓₓ * bc * v̂ - f.(x)

    # Find compensated initial condition 
    v̂₀ = linear_interpolation(x,nlsolve(func,v₀.(x)).zero,extrapolation_bc=Line())

    #! check γ
    û,v̂ = (x -> n₀(x) - v̂₀(x)),v̂₀
    γ̂ = x -> D₁ * ∂ₓₓ(û,x) + α₁ * ∂ₓ(û,x) + p̂₁ * û(x) + p̂₂ * v̂(x) .+ p̂₃ + 
             D₂ * ∂ₓₓ(v̂,x) + α₂ * ∂ₓ(v̂,x) + p̂₄ * û(x) + p̂₅ * v̂(x) .+ p̂₆
    plot(γ,xlim=(0.0,100.0))
    plot!(γ̂,xlim=(0.0,100.0))

## Plot curves with compensated parameters
sol,x = solve_model(p̂,v̂₀;N);
pu₂ = plot(legend=:none)
pv₂ = plot(legend=:none)
for t = 0.0:2.0:6.0
    u,v = sol(t)[1:length(x)],sol(t)[length(x)+1:end]
    w = u + v
    plot!(pu₂,x,u,lw=2.0,c=:red,ls=:dash)
    plot!(pv₂,x,v,lw=2.0,c=:red,ls=:dash)
    plot!(pn,x,w,lw=2.0,c=:red,ls=:dash)
end
plot(pu₂,pv₂,pn)

#######################################
## FIGURE
#######################################

p1 = plot(pu₁,pv₁,pu₂,pv₂,ylim=(0.0,15.0))
plot!(p1,subplot=2,ylim=(0.0,8.0))
plot!(p1,subplot=4,ylim=(0.0,8.0))
plot!(p1,subplot=1,xticks=(0:25:100,[]),bottom_margin=-2Plots.mm,ylabel="u(x,t)")
plot!(p1,subplot=2,xticks=(0:25:100,[]),bottom_margin=-2Plots.mm,ylabel="v(x,t)")
plot!(p1,subplot=3,xlabel="x",ylabel="u(x,t)")
plot!(p1,subplot=4,xlabel="x",ylabel="v(x,t)")

p2 = plot(pn,ylim=(0.0,20.0),xlabel="x",widen=true)
p3 = plot(pltode,ylim=(0.0,50.0),xlabel="t",widen=true)

fig1 = plot(p1,p2,p3,layout=@layout([a{0.37w} b c]),size=(800,220))
savefig("fig1.svg")