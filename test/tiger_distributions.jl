begin
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    using CGCP
    Pkg.activate(@__DIR__)
    using POMDPs
    using POMDPTools
    using POMDPModels
    using ConstrainedPOMDPs
    using ConstrainedPOMDPModels
    using ProgressMeter
end


cpomdp = constrain(TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.75), [3.0]) do s,a
    iszero(a) ? [1.0] : [0.0]
end

sol = CGCPSolver()
pol = solve(sol, cpomdp)

N = 100_000
sim_rewards = zeros(N)
sim_costs = zeros(N)

@showprogress for i ∈ 1:N
    p_idx = rand(SparseCat(eachindex(pol.policy_vector), pol.p_pi))
    policy = pol.policy_vector[p_idx]
    sim = RolloutSimulator(max_steps = 20)
    r,c = simulate(sim, cpomdp, policy, DiscreteUpdater(cpomdp), initialstate(cpomdp))
    sim_rewards[i] = r
    sim_costs[i] = only(c)
end

using Plots
using StatsPlots

@show mean(sim_costs)
@show mean(sim_rewards)

default(grid=false, framestyle=:box, fontfamily="Computer Modern", label="")

p1 = violin([""], sim_rewards; ylabel="reward", show_mean=true, bandwidth=1.0)
p2 = violin([""], sim_costs; ylabel="cost", c=:red, show_mean=true, bandwidth=0.05)
p = plot(p1,p2, dpi=300)

savefig(p, "CGCP_violin.png")
