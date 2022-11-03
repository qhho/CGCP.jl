macro until(condition, expression)
    quote
        while !($condition)
            $expression
        end
    end |> esc
end

## Solver Struct for POMDPs.jl style
struct CGCPSolver
    max_time::Float64
    τ_inc::Float64
    ρ::Float64
end

function CGCPSolver(; max_time=1e5, τ_inc=100.0, ρ=3.0)
    return CGCPSolver(max_time,τ_inc,ρ)
end

mutable struct ProbWrap
    m::ConstrainedPOMDPWrapper
    λ::Vector{Float64}
    initialized::Bool
end

##need to make this part of core instead of in this file
function POMDPs.reward(m::ConstrainedPOMDPs.ConstrainedPOMDPWrapper, s, a)
    return m.initialized*reward(m.m, s, a) - m.λ*ConstrainedPOMDPs.cost(m.m,s,a)
end

function StateActionReward(m)
    return FunctionSARC(m)
end

struct FunctionSARC{M} <: POMDPTools.StateActionReward
    m::M
end

function (sarc::FunctionSARC)(s, a)
    if isterminal(sarc.m, s)
        return 0.0
    else
        return m.initialized*reward(sarc.m.m, s, a) - sarc.m.λ*ConstrainedPOMDPs.cost(sarc.m, s, a)
    end
end

function initialize_master(m::ProbWrap, solver, sim::ConstrainedPOMDPs.RolloutSimulator)
    #check correctness of this
    policy_vector = Vector{AlphaVectorPolicy}(undef, 0)
    mlp = Model(GLPK.Optimizer)
    @variable(mlp, x[1:1] >= 0)
    λ = ones(length(m.m.constraints))
    τ = 20.0
    m.initialized = false
    policy = compute_policy(m, solver, λ, τ, 1.0)
    push!(policy_vector, policy)
    v, c = evaluate_policy(m, policy, sim)
    m.initialized = true
    @objective(
            mlp,
            Max,
            sum(v*x[i] for i in 1:1)
        )

    @constraint(mlp, dualcon, sum(c*x[1]) <= m.m.constraints[1])
    @constraint(mlp, validprobability, sum(x[i] for i in 1:1) == 1.0)

    return mlp, x, dualcon, validprobability, policy_vector
end

function add_column_to_master!(mlp, x, v, c, dualcon, validprobability, ncols)
    #check correctness of this
    push!(x, @variable(mlp, base_name = "x[" * string(ncols) * "]", lower_bound = 0.0))
    set_objective_coefficient(
        mlp,
        x[ncols],
        v,
        )
    set_normalized_coefficient(dualcon, x[ncols], c)
    set_normalized_coefficient(validprobability, x[ncols], 1)
end

function evaluate_policy(m::ProbWrap, policy, simmer::ConstrainedPOMDPs.RolloutSimulator; parallel=true)
    #monte carlo simulation evaluation of policy
    #TODO: currently, we use MC evaluation instead of policy graph with lots of
    # need to use policy graph evaluation for comparison
    n_sim = 100
    total_v = 0.0
    total_c = 0.0
    λ = m.λ
    m.λ = zeros(length(m.m.constraints))
    m.initialized = true
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n_sim
            v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, updater(policy), initialstate(m.m), rand(initialstate(m.m)))
            total_v += v
            total_c += c
        end
    else
        for i in 1:n_sim
            v, c = ConstrainedPOMDPs.simulate(simmer, m.m, policy, updater(policy), initialstate(m.m), rand(initialstate(m.m)))
            total_v += v
            total_c += c
        end
    end
    m.λ = λ
    return total_v/n_sim, ceil((total_c/n_sim),digits = 3)
end

function compute_policy(m::ProbWrap, solver, λ::Vector{Float64}, τ::Float64, ρ::Float64)
    m.λ = λ
    solver.timeout = τ
    return SARSOP.solve(solver, m.m)
end

function POMDPs.solve(solver::CGCPSolver, pomdp::ConstrainedPOMDPWrapper)
    M = ProbWrap(pomdp,ones(length(pomdp.constraints)),false)
    max_time = solver.max_time
    τ_inc = solver.τ_inc
    ρ = solver.ρ

    #check correctness of this
    solver = SARSOPSolver(precision=1e-3, verbose= true, fast = true)
    simmer = ConstrainedPOMDPs.RolloutSimulator(max_steps = 500, rng=Xoroshiro128Plus(1))
    mlp, x, dualcon, validprobability, policy_vector = initialize_master(M, solver, simmer)

    # T_p = 0.0
    dual_vectors = [Inf]
    λ_p = Inf
    τ = 10.0
    ncols = 1
    t_0 = time()
    λ = 1.0
    @until time() - t_0 >= max_time begin
        optimize!(mlp)
        λ = shadow_price(dualcon)
        ncols += 1
        @show (time() - t_0)

        push!(dual_vectors, λ)
        if λ == λ_p
            τ += τ_inc
        end
        t_temp = time()
        policy = compute_policy(M, solver, λ, τ,ρ)
        @show time(), t_temp
        t_0 += (time() - t_temp) - τ
        V, C = evaluate_policy(M, policy, simmer)
        add_column_to_master!(mlp, x, V, C, dualcon, validprobability, ncols)
        # ϕ_upper += policy_upper_bound
        push!(policy_vector , policy)
        λ_p = λ
        @show (time() - t_0)
    end

    optimize!(mlp)

    p_pi = JuMP.value.(x)
    return policy_vector, p_pi, mlp, dual_vectors
end
