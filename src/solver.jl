Base.@kwdef struct CGCPSolver{LP, EVAL, O<:NamedTuple}
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    max_steps::Int      = typemax(Int)
    τ::Float64          = 20.0
    τ_inc::Float64      = 100.0
    ρ::Float64          = 3.0
    lp_solver::LP       = GLPK.Optimizer
    evaluator::EVAL     = PolicyGraphEvaluator(max_steps) #MCEvaluator()
    verbose::Bool       = false
    pomdp_sol_options::O= (;delta=0.75)
end

mutable struct CGCPSolution <: Policy
    const policy_vector::Vector{AlphaVectorPolicy}
    const p_pi::Vector{Float64}
    const mlp::JuMP.Model
    const C::Matrix{Float64}
    const V::Vector{Float64}
    const dual_vectors::Vector{Vector{Float64}}
    policy_idx::Int
    const problem::CGCPProblem
    const evaluator::Union{PolicyGraphEvaluator,RecursiveEvaluator,MCEvaluator}
end

##
function initial_policy(sol::CGCPSolver, m::CGCPProblem, pomdp_solver)
    m.initialized = false

    λ = ones(length(m.m.constraints))
    policy = compute_policy(pomdp_solver, m, λ)
    v, c = evaluate_policy(sol.evaluator, m, policy)

    m.initialized = true
    return policy, v, c
end

function compute_policy(sol, m::CGCPProblem, λ::Vector{Float64})
    m.λ = λ
    s_pol = solve(sol,m) 
    return  s_pol
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

function POMDPs.solve(solver::CGCPSolver, pomdp::CPOMDP)
    t0 = time()
    (;max_time, max_iter, evaluator, verbose, τ, pomdp_sol_options) = solver
    nc = constraint_size(pomdp)
    prob = CGCPProblem(pomdp, ones(nc), false)

    pomdp_solver = HSVI4CGCP.SARSOPSolver(;max_time=τ, max_steps=solver.max_steps, pomdp_sol_options...)
    π0, v0, c0 = initial_policy(solver, prob, pomdp_solver)
    Π = [π0]
    V = [v0]
    C = reshape(c0, nc, 1)
    
    lp = master_lp(solver, prob, C,V)
    
    # if cost minimizing policy inadmissible, no point in solving LP or finding other solutions
    # Just return min cost solution
    if !(c0 ⪯ constraints(pomdp))
        return CGCPSolution(Π, [1.0], lp, C, V, ones(nc), 0, prob, evaluator)
    end

    optimize!(lp)
    λ = dual(lp[:CONSTRAINT])::Vector{Float64}
    λ_hist = [λ]

    πt = compute_policy(pomdp_solver,prob,λ)
    v_t, c_t = evaluate_policy(evaluator, prob, πt)

    iter = 1
    verbose && println("""
        iteration $iter
        c = $c_t
        v = $v_t
        λ = $λ
    ----------------------------------------------------
    """)
    
    while time() - t0 < max_time && iter < max_iter
        push!(Π, πt)
        push!(V, v_t)
        C = hcat(C, c_t)
        iter += 1

        lp = master_lp(solver,prob,C,V)
        optimize!(lp)
        λ = dual(lp[:CONSTRAINT])::Vector{Float64}
        push!(λ_hist, λ)

        ϕl = JuMP.objective_value(lp) 
        ϕu = dot(λ,constraints(pomdp))

        if λ == λ_hist[end-1]
            τ += solver.τ_inc
        end 

        δ = maximum(abs, λ .- λ_hist[end-1])
        
        pomdp_solver = HSVI4CGCP.SARSOPSolver(max_time=τ,max_steps=solver.max_steps,delta=0.75)
        πt = compute_policy(pomdp_solver,prob,λ)
        v_t, c_t = evaluate_policy(evaluator, prob, πt)
        ϕu += POMDPs.value(πt,initialstate(pomdp))
        ϕa = 10^(log10(max(abs(ϕl),abs(ϕu)))-solver.ρ)

        verbose && println("""
            iteration $iter
            c = $c_t
            v = $v_t
            λ = $λ
            δ = $δ
            ϕa = $ϕa
            Δϕ = $(ϕu-ϕl)
        ----------------------------------------------------
        """)
        ((ϕu-ϕl)<ϕa) && break
    end
    return CGCPSolution(Π, JuMP.value.(lp[:x]), lp, C, V, λ_hist, 0, prob, evaluator)
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


function POMDPs.value(p::CGCPSolution, b)
    return iszero(p.policy_idx) ? probabilistic_value(p, b) : deterministic_value(p, b)
end

function probabilistic_value(p::CGCPSolution, b)
    v = 0.0
    for (π_i, p_i) ∈ zip(p.policy_vector, p.p_pi)
        v += p_i*evaluate_policy(p.evaluator, p.problem, π_i, b)[1]
    end
    return v
end

function deterministic_value(p::CGCPSolution, b)
    @assert !iszero(p.policy_idx)
    return evaluate_policy(p.evaluator, p.problem, p.policy_vector[p.policy_idx], b)[1]
end

function lagrange_probabilistic_value(p::CGCPSolution, b)
    v = 0.0
    for (π_i, p_i) ∈ zip(p.policy_vector, p.p_pi)
        v += p_i*POMDPs.value(π_i, b)
    end
    return v
end

function lagrange_deterministic_value(p::CGCPSolution, b)
    @assert !iszero(p.policy_idx)
    return POMDPs.value(p.policy_vector[p.policy_idx], b)
end
