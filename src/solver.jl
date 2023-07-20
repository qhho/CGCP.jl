Base.@kwdef struct CGCPSolver{LP, P, EVAL}
    max_time::Float64   = 1e5
    max_iter::Int       = 100
    τ_inc::Float64      = 100.0
    ρ::Float64          = 3.0
    lp_solver::LP       = GLPK.Optimizer
    pomdp_solver::P     = SARSOP.SARSOPSolver(precision=1.0,verbose=false,timeout=5.0)#NativeSARSOP.SARSOPSolver(verbose=false,precision=1.0,delta=1.0,max_time=5.0,epsilon=1.0)#SARSOP.SARSOPSolver(precision=1.0,verbose=false,timeout=5.0) #NativeSARSOP.SARSOPSolver(verbose=false,precision=1.0,delta=0.2,max_time=0.5)#PBVISolver(max_time=5.0,max_iter=typemax(Int))
    ϵ::Float64          = 1e-3
    evaluator::EVAL     = MCEvaluator() #PolicyGraphEvaluator() #MCEvaluator()
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
    const problem::CGCPProblem
    const evaluator::Union{PolicyGraphEvaluator,RecursiveEvaluator,MCEvaluator}
end

##
function initial_policy(sol::CGCPSolver, m::CGCPProblem, pomdp_solver)
    m.initialized = false

    # m.initialized = true
    # λ = zeros(length(m.m.constraints))

    λ = ones(length(m.m.constraints))
    policy = compute_policy(pomdp_solver, m, λ)
    # @show m
    # @show evaluate_policy(PolicyGraphEvaluator(;method=belief_value_recursive), deepcopy(m), deepcopy(policy))
    # @show evaluate_policy(MCEvaluator(), deepcopy(m), deepcopy(policy))
    # @show m
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
    (;max_time, max_iter, evaluator, verbose) = solver
    nc = constraint_size(pomdp)
    prob = CGCPProblem(pomdp, ones(nc), false)

    τ = 5.0 #solver.τ 
    pomdp_solver = NativeSARSOP.SARSOPSolver(max_time=10.0)#PBVISolver(max_time=0.0325,max_iter=typemax(Int))#NativeSARSOP.SARSOPSolver(ρ=3.0) #PBVISolver(max_time=0.0325,max_iter=typemax(Int))#SARSOP.SARSOPSolver(precision=1.0,verbose=false,timeout=τ)
    π0, v0, c0 = initial_policy(solver, prob, pomdp_solver)
    # @show "initial_policy done"
    Π = [π0]
    V = [v0]
    C = reshape(c0, nc, 1)

    lp = master_lp(solver, prob, C,V)
    optimize!(lp)
    λ = dual(lp[:CONSTRAINT])::Vector{Float64}
    λ_hist = [λ]
    # @show λ_hist
    # @show "LP done"

    πt = compute_policy(pomdp_solver,prob,λ)
    # @show "Compute done"
    # return πt
    v_t, c_t = evaluate_policy(evaluator, prob, πt)
    # @show "Evaluate done"
    @show λ
    iter = 1
    while time() - t0 < max_time && iter < max_iter
        # @show iter
        push!(Π, πt)
        push!(V, v_t)
        C = hcat(C, c_t)
        iter += 1

        lp = master_lp(solver,prob,C,V)
        optimize!(lp)
        λ = dual(lp[:CONSTRAINT])::Vector{Float64}
        push!(λ_hist, λ)
        # @show "LP done"
        @show λ

        ϕl = JuMP.objective_value(lp) 
        ϕu = dot(λ,constraints(pomdp))

        if λ == λ_hist[end-1]
            τ += solver.τ_inc
        end 

        δ = maximum(abs, λ .- λ_hist[end-1])
        # δ < solver.ϵ && break
        
        pomdp_solver = NativeSARSOP.SARSOPSolver(max_time=10.0) #PBVISolver(max_time=0.0325,max_iter=typemax(Int))#NativeSARSOP.SARSOPSolver(ρ=3.0)#PBVISolver(max_time=0.0325,max_iter=typemax(Int))#SARSOP.SARSOPSolver(precision=1.0,verbose=false,timeout=τ)
        πt = compute_policy(pomdp_solver,prob,λ)
        # @show λ
        # @show "Compute done"
        # return πt
        v_t, c_t = evaluate_policy(evaluator, prob, πt)
        # @show "Evaluate done"
        ϕu += POMDPs.value(πt,initialstate(pomdp))
        ϕa = 10^(log10(max(abs(ϕl),abs(ϕu)))-solver.ρ)
        verbose && println("""
            c = $c_t
            v = $v_t
            λ = $λ
            δ = $δ
            ϕa = $ϕa
            Δϕ = $(ϕu-ϕl)
        ----------------------------------------------------
        """)
        (ϕu-ϕl<ϕa) && break
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