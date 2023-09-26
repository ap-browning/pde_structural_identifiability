gr()
default()
default(
    fontfamily="Helvetica",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Helvetica",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Arial"
    )
end

# Colours
col_prior1 = "#2d79ff"
col_prior2 = "#af52de"
col_posterior = "#f63a30"


using KernelDensity
using Plots, StatsPlots

"""
    density2d(x,y,...)

Create 2D kernel density plot.
"""
@userplot density2d
@recipe function f(kc::density2d; levels=10, clip=((-3.0, 3.0), (-3.0, 3.0)), z_clip = nothing)
    x,y = kc.args

    x = vec(x)
    y = vec(y)

    k = KernelDensity.kde((x, y))
    z = k.density / maximum(k.density)
    if !isnothing(z_clip)
        z[z .< z_clip * maximum(z)] .= NaN
    end

    legend --> false

    @series begin
        seriestype := contourf
        colorbar := false
        (collect(k.x), collect(k.y), z')
    end

end


function maximise(f,x₀;lb=fill(-Inf,length(x₀)),ub=fill(-Inf,length(x₀)),alg=:LN_NELDERMEAD,autodiff=false,ftol_abs=1e-8,maxtime=30)
    function func(x::Vector,dx::Vector)
        length(dx) > 0 && autodiff == :forward && copyto!(dx,ForwardDiff.gradient(f,x))
        return f(x)
    end
    opt = NLopt.Opt(alg,length(x₀))
    opt.max_objective = func
    opt.maxtime = maxtime
    opt.ftol_abs = ftol_abs
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    (minf,minx,ret) = NLopt.optimize(opt,x₀)
    return minf,minx
end

using AdaptiveMCMC
using MCMCChains
using .Threads
function mcmc(logpost,x₀,iters;algorithm=:aswam,nchains=1,L=1,burnin=1,thin=1,param_names=nothing,parallel=false)

    # Perform MCMC
    res = Array{Any}(undef,nchains)
    if parallel
        @threads for i = 1:nchains
            res[i] = adaptive_rwm(x₀,logpost,iters;L,algorithm,thin,b=burnin)
        end
    else
        for i = 1:nchains
            res[i] = adaptive_rwm(x₀,logpost,iters;L,algorithm,thin,b=burnin)
        end
    end

    # Create MCMCChains object
    Chains(
        cat([r.X' for r in res]...,dims=3),
        param_names === nothing ? ["p$i" for i = 1:size(res[1].X,1)] : param_names;
        evidence = cat([r.D[1]' for r in res]...,dims=3),
        start=burnin+1,
        thin=thin,
        iterations=burnin:thin:iters
    )
end