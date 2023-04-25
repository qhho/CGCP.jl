Base.@kwdef struct MCEvaluator
    n::Int         = 100
    parallel::Bool = false
    max_steps::Int = 100
end

function evaluate_policy(eval::MCEvaluator, m::CGCPProblem, policy)
    return if eval.parallel && Threads.nthreads() > 1
        parallel_evaluate(eval, m, policy)
    else
        serial_evaluate(eval, m, policy)
    end
end

function serial_evaluate(eval::MCEvaluator, m::CGCPProblem, policy)
    (;n, max_steps) = eval
    total_v = 0.0
    total_c = zeros(constraint_size(m.m))
    for i in 1:n
        v, c = ConstrainedPOMDPs.simulate(
            RolloutSimulator(;max_steps, rng=Random.default_rng()),
            m.m, policy, 
            DiscreteUpdater(m.m.m), 
            initialstate(m), 
            rand(initialstate(m))
        )
        total_v += v
        total_c .+= c
    end
    v̂ = total_v / n
    ĉ = total_c ./ n
    return v̂, ĉ
end

function parallel_evaluate(eval::MCEvaluator, m::CGCPProblem, policy)
    (;n, max_steps) = eval
    total_v = 0.0
    total_c = zeros(constraint_size(m.m))
    Threads.@threads for i in 1:n
        v, c = ConstrainedPOMDPs.simulate(
            RolloutSimulator(;max_steps, rng=Random.default_rng()),
            m.m, policy, 
            DiscreteUpdater(m.m.m), 
            initialstate(m), 
            rand(initialstate(m))
        )
        total_v += v
        total_c .+= c
    end
    v̂ = total_v / n
    ĉ = total_c ./ n
    return v̂, ĉ
end

# function evaluate_policy(m::CGCPProblem, policy)
#     #monte carlo simulation evaluation of policy
#     #TODO: currently, we use MC evaluation instead of policy graph with lots of
#     # need to use policy graph evaluation for comparison
#     n_sim = 100
#     total_v = 0.0
#     total_c = zeros(constraint_size(m.m))
#     if parallel && Threads.nthreads() > 1
#         # FIXME: All threads should NOT be using the same simulator and working off of the same rng - also fix race condition
#         Threads.@threads for i in 1:n_sim 
#             v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, DiscreteUpdater(m.m.m), initialstate(m), rand(initialstate(m)))
#             total_v += v
#             total_c .+= c
#         end
#     else
#         for i in 1:n_sim
#             v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, DiscreteUpdater(m.m.m), initialstate(m), rand(initialstate(m)))
#             total_v += v
#             total_c .+= c
#         end
#     end
#     v̂ = total_v / n_sim
#     ĉ = total_c ./ n_sim
#     return v̂, ĉ
# end
