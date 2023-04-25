Base.@kwdef struct CGCPSolver{LP, P, EVAL}
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    τ_inc::Float64      = 100.0
    ρ::Float64          = 3.0
    lp_solver::LP       = GLPK.Optimizer
    pomdp_solver::P     = PBVISolver(max_iter=5)
    ϵ::Float64          = 1e-3
    evaluator::EVAL     = MCEvaluator()
end

struct CGCPSolution <: Policy
    policy_vector::Vector{POMDPTools.Policies.AlphaVectorPolicy}
    p_pi::Vector{Float64}
    mlp::JuMP.Model
    C::Matrix{Float64}
    V::Vector{Float64}
    dual_vectors::Vector{Vector{Float64}}
end

##
function initial_policy(sol::CGCPSolver, m::CGCPProblem)
    m.initialized = false
    λ = ones(length(m.m.constraints))
    policy = compute_policy(sol, m, λ)
    v, c = evaluate_policy(sol.evaluator, m, policy)
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
    @constraint(lp, CONSTRAINT, C*x ≤ Ĉ)
    @constraint(lp, SIMPLEX, sum(x) == 1.0)
    return lp
end

function POMDPs.solve(solver::CGCPSolver, pomdp::ConstrainedPOMDPWrapper)
    t0 = time()
    (;max_time, max_iter, evaluator) = solver
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
        maximum(abs, λ .- λ_hist[end-1]) < solver.ϵ && break
    end
    return CGCPSolution(Π, JuMP.value.(lp[:x]), lp, C, V, λ_hist)
end

function POMDPs.action(policy::CGCPSolution, x)
    return action(last(policy.policy_vector), x)
end

function POMDPs.value(policy::CGCPSolution, s)
    return value(last(policy.policy_vector), s)
end

function POMDPs.value(policy::CGCPSolution, s, a)
    return value(last(policy.policy_vector), s, a)
end
