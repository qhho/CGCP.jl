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

# FIXME: make threadsafe
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

##
Base.@kwdef struct PolicyGraphEvaluator
    h::Int = typemax(Int) # seems a bit excessive
    method::Function = recursive_evaluation
end

function evaluate_policy(eval::PolicyGraphEvaluator, m::CGCPProblem, policy)
    up = DiscreteUpdater(m.m.m)
    b0 = initialize_belief(up,initialstate(m))
    v,c... = eval.method(m, up, policy, b0, eval.h; rewardfunction=PG_reward)
    return v,c
end
