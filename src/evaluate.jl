Base.@kwdef struct MCEvaluator
    n::Int         = 1000 #100
    parallel::Bool = false
    max_steps::Int = 1000 #100
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
            DiscreteUpdater(m.m), 
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
    total_v = zeros(n)
    total_c = [zeros(constraint_size(m.m)) for _ in 1:n]
    Threads.@threads for i in 1:n
        v, c = ConstrainedPOMDPs.simulate(
            RolloutSimulator(;max_steps, rng=Random.default_rng()),
            m.m, policy, 
            DiscreteUpdater(m.m), 
            initialstate(m), 
            rand(initialstate(m))
        )
        total_v[i] = v
        total_c[i] .= c
    end
    v̂ = mean(total_v)
    ĉ = mean(total_c)
    return v̂, ĉ
end

function evaluate_policy(eval::MCEvaluator, m::CGCPProblem, policy, b)
    up = DiscreteUpdater(m.m)
    b0 = initialize_belief(up,b)
    return  evaluate_policy(eval, m, policy, b0)
end

function evaluate_policy(eval::MCEvaluator, m::CGCPProblem, policy, b::DiscreteBelief)
    return if eval.parallel && Threads.nthreads() > 1
        parallel_evaluate(eval, m, policy, b)
    else
        serial_evaluate(eval, m, policy, b)
    end
end

function serial_evaluate(eval::MCEvaluator, m::CGCPProblem, policy, b)
    (;n, max_steps) = eval
    total_v = 0.0
    total_c = zeros(constraint_size(m.m))
    for i in 1:n
        v, c = ConstrainedPOMDPs.simulate(
            RolloutSimulator(;max_steps, rng=Random.default_rng()),
            m.m, policy, 
            DiscreteUpdater(m.m), 
            b, 
            rand(b)
        )
        total_v += v
        total_c .+= c
    end
    v̂ = total_v / n
    ĉ = total_c ./ n
    return v̂, ĉ
end

# FIXME: make threadsafe
function parallel_evaluate(eval::MCEvaluator, m::CGCPProblem, policy, b)
    (;n, max_steps) = eval
    total_v = 0.0
    total_c = zeros(constraint_size(m.m))
    Threads.@threads for i in 1:n
        v, c = ConstrainedPOMDPs.simulate(
            RolloutSimulator(;max_steps, rng=Random.default_rng()),
            m.m, policy, 
            DiscreteUpdater(m.m), 
            b, 
            rand(b)
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
    h::Int = 100 #typemax(Int) # seems a bit excessive
    method::Function = belief_value_polgraph
end

function evaluate_policy(eval::PolicyGraphEvaluator, m::CGCPProblem, policy)
    up = DiscreteUpdater(m.m)
    b0 = initialize_belief(up,initialstate(m))
    v,c... = eval.method(m, up, policy, b0, eval.h; rewardfunction=PG_reward)
    return v,c
end

function evaluate_policy(eval::PolicyGraphEvaluator, m::CGCPProblem, policy, b)
    up = DiscreteUpdater(m.m)
    b0 = initialize_belief(up,b)
    v,c... = eval.method(m, up, policy, b0, eval.h; rewardfunction=PG_reward)
    return v,c
end
