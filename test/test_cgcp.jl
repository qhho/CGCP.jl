using ConstrainedPOMDPs
using POMDPs
using POMDPModels
using POMDPTools
using Random

pomdp = TigerPOMDP()
m = Constrain(pomdp, [1.0])

# simmer = RolloutSimulator(max_steps = 100, rng=Random.MersenneTwister(1))

# solver = PBVISolver()
# policy = solve(solver, m)
# evaluate_policy(m, policy, simmer)

function ConstrainedPOMDPs.cost(m, s, a)
    if a == 0
        return [10.0]
    elseif a == 2
        return [0.0]
    else
        return [0.0]
    end
end

# policy_vector, p_pi, mlp, dual_vectors = solve(CGCPSolver(),m) #CGCP(m, 10.0, 1, 0.5)

# print(mlp)
# @show p_pi
# @show dual_vectors

# @show termination_status(mlp)