using CGCP
# include("../ConstrainedPOMDPs.jl")
using ConstrainedPOMDPs
using POMDPs
using POMDPModels
using POMDPModelTools
using Random
using RockSample

m = RockSamplePOMDP(5, 7)

function ConstrainedPOMDPs.costs(m, s, a)
    # @show a
    # @show reward(m, s, a)
    if a > 5
        return 1.0
    elseif reward(m,s, a) < 0.0
        return 1.0
    end
    return 0.0
    # return 1
end

cm = Constrain(m, [1.0])
# 
policy_vector, p_pi, mlp, dual_vectors = CGCP(cm, 30.0, 0.1, 0.5)

# include("../../models/CCPBVI_Toy.jl")

# policy_vector, p_pi, mlp, dual_vectors = CGCP(cm, 1.0, 0.1, 0.5)

# print(mlp)
# @show p_pi
# @show dual_vectors

# @show termination_status(mlp)

indices = findall(>(0), p_pi)

simmer = RolloutSimulator(max_steps = 1000, rng=Random.MersenneTwister(1))

output = []
Threads.@threads for i in 1:100
    index = -1
    p = 0
    randp = rand()
    for (ind, prob) in enumerate(p_pi)
        # @show prob
        randp -= prob
        if randp < 0
            index = ind
            break
        end
    end
    # policy = policy_vector[index]
    policy = policy_vector[index]
    # policy = policydeterministic_57
    push!(output, [ConstrainedPOMDPs.simulate(simmer, cm, policy, updater(policy), initialstate(cm), rand(initialstate(cm))), index])
end

mean(output)

# using POMDPGifs

# sim = GifSimulator(filename="test.gif", max_steps=30)

# policy = policy_vector[3]

# simulate(sim, cm, policy, updater(policy), initialstate(cm), rand(initialstate(cm))