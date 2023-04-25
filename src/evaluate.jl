Base.@kwdef struct MCEvaluator
    n::Int         = 100
    parallel::Bool = false
end

function evaluate_policy(m::CGCPProblem, policy, simmer::ConstrainedPOMDPs.RolloutSimulator; parallel=true)
    #monte carlo simulation evaluation of policy
    #TODO: currently, we use MC evaluation instead of policy graph with lots of
    # need to use policy graph evaluation for comparison
    n_sim = 100
    total_v = 0.0
    total_c = zeros(constraint_size(m.m))
    λ = m.λ
    @show λ
    m.λ = zeros(length(m.m.constraints))
    m.initialized = true
    if parallel && Threads.nthreads() > 1
        # FIXME: All threads should NOT be using the same simulator and working off of the same rng - also fix race condition
        Threads.@threads for i in 1:n_sim 
            v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, DiscreteUpdater(m.m.m), initialstate(m), rand(initialstate(m)))
            total_v += v
            total_c += c
        end
    else
        for i in 1:n_sim
            v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, DiscreteUpdater(m.m.m), initialstate(m), rand(initialstate(m)))
            total_v += v
            total_c += c
        end
    end
    m.λ = λ
    @show total_v/n_sim, ceil.((total_c/n_sim),digits = 3)

    up = DiscreteUpdater(m)
    b0 = initialize_belief(up,initialstate(m))
    # pg = GenandEvalPG(m,up,policy,b0,5;rewardfunction=PG_reward)
    pg_val = BeliefValue(m,up,policy,b0,5;rewardfunction=PG_reward)
    v = pg_val[1]
    c = ceil.(pg_val[2:end],digits = 3)
    @show v,c
    return v,c
end
