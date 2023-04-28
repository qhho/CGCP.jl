Base.@kwdef struct CGCPSolver{LP, P, EVAL}
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    τ_inc::Float64      = 100.0
    ρ::Float64          = 3.0
    lp_solver::LP       = GLPK.Optimizer
    pomdp_solver::P     = PBVISolver(max_iter=5)
    ϵ::Float64          = 1e-3
    evaluator::EVAL     = MCEvaluator()
    verbose::Bool       = false
end

mutable struct CGCPSolution <: Policy
    const policy_vector::Vector{AlphaVectorPolicy}
    const p_pi::Vector{Float64}
    const mlp::JuMP.Model
    const C::Matrix{Float64}
    const V::Vector{Float64}
    const dual_vectors::Vector{Vector{Float64}}
    policy_idx::Int
end

##
function initial_policy(sol::CGCPSolver, m::CGCPProblem)
    m.initialized = false
    λ = ones(length(m.m.constraints))
    policy = compute_policy(sol, m, λ)
    v, c = evaluate_policy(sol.evaluator, m, policy)
    m.initialized = true
    return policy, v, c
end

function compute_policy(sol::CGCPSolver, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    return solve(sol.pomdp_solver, m)
end

function master_lp(solver::CGCPSolver, m::CGCPProblem, C, V)
    lp = Model(solver.lp_solver)
    n = length(V)
    Ĉ = constraints(m.m)
    @variable(lp, x[1:n] ≥ 0)
    @objective(lp, Max, dot(V, x))

    # NOTE: constraint this way around keeps λ ≥ 0???
    # using C*x ≤ Ĉ causes λ ≤ 0
    @constraint(lp, CONSTRAINT, Ĉ ≥ C*x) 
    @constraint(lp, SIMPLEX, sum(x) == 1.0)
    return lp
end

function POMDPs.solve(solver::CGCPSolver, pomdp::ConstrainedPOMDPWrapper)
    t0 = time()
    (;max_time, max_iter, evaluator, verbose) = solver
    nc = constraint_size(pomdp)
    prob = CGCPProblem(pomdp, ones(nc), false)
    π0, v0, c0 = initial_policy(solver, prob)
    Π = [π0]
    V = [v0]
    C = reshape(c0, nc, 1)

    lp = master_lp(solver, prob, C,V)
    optimize!(lp)
    λ = dual(lp[:CONSTRAINT])
    λ_hist = [λ]

    iter = 0
    while time() - t0 < max_time && iter < max_iter
        iter += 1

        πt = compute_policy(solver,prob, λ)
        v_t, c_t = evaluate_policy(evaluator, prob, πt)
        push!(Π, πt)
        push!(V, v_t)
        C = hcat(C, c_t)

        lp = master_lp(solver,prob,C,V)
        optimize!(lp)
        λ = dual(lp[:CONSTRAINT])
        push!(λ_hist, λ)
        δ = maximum(abs, λ .- λ_hist[end-1])
        verbose && println("""
            c = $c_t
            v = $v_t
            λ = $λ
            δ = $δ
        ----------------------------------------------------
        """)
        δ < solver.ϵ && break
    end
    return CGCPSolution(Π, JuMP.value.(lp[:x]), lp, C, V, λ_hist, 0)
end

reset!(p::CGCPSolution) = p.policy_idx = 0

function initialize!(p::CGCPSolution)
    probs = p.p_pi
    p.policy_idx = rand(SparseCat(eachindex(probs), probs))
    p
end

function POMDPs.action(p::CGCPSolution, b)
    iszero(p.policy_idx) && initialize!(p)
    return action(p.policy_vector[p.policy_idx], b)
end

# FIXME: this is using lagrangian values, not pure constrained reward values
function POMDPs.value(p::CGCPSolution, b)
    return iszero(p.policy_idx) ? probabilistic_value(p, b) : deterministic_value(p, b)
end

function probabilistic_value(p::CGCPSolution, b)
    v = 0.0
    for (π_i, p_i) ∈ zip(p.policy_vector, p.p_pi)
        v += p_i*POMDPs.value(π_i, b)
    end
    return v
end

function deterministic_value(p::CGCPSolution, b)
    @assert !iszero(p.policy_idx)
    return POMDPs.value(p.policy_vector[p.policy_idx], b)
end
