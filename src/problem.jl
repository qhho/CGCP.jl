mutable struct CGCPProblem{S,A,O,M<:POMDP} <: POMDP{Tuple{S, Int}, A, Tuple{O,Int}}
    _pomdp::M
    m::ConstrainedPOMDPWrapper{S,A,O,M}
    λ::Vector{Float64}
    initialized::Bool
end

function CGCPProblem(m::ConstrainedPOMDPWrapper, λ::Vector{Float64}, initialized::Bool)
    return CGCPProblem{statetype(m.m), actiontype(m.m), obstype(m.m), typeof(m.m)}(m.m, m, λ, initialized)
end

##Functions for Simulating Weighted Sums

POMDPs.states(m::CGCPProblem) = states(m.m)
POMDPs.actions(w::CGCPProblem) = actions(w.m)
POMDPs.observations(w::CGCPProblem) = observations(w.m)
POMDPs.actionindex(w::CGCPProblem, a) = actionindex(w.m, a)
POMDPs.discount(w::CGCPProblem) = discount(w.m)
POMDPs.stateindex(w::CGCPProblem, s) = stateindex(w.m, s)
POMDPs.statetype(m::CGCPProblem) = statetype(m.m)
POMDPs.actiontype(m::CGCPProblem) = actiontype(m.m)
POMDPs.obstype(m::CGCPProblem) = obstype(m.m)
POMDPs.initialstate(m::CGCPProblem) = initialstate(m.m)
POMDPs.obsindex(m::CGCPProblem, o) = obsindex(m.m, o)
POMDPs.transition(m::CGCPProblem, s, a) = transition(m.m, s, a)
POMDPs.observation(m::CGCPProblem, a, s) = observation(m.m, a, s)
POMDPTools.ordered_states(m::CGCPProblem) = ordered_states(m.m)
POMDPTools.ordered_actions(m::CGCPProblem) = ordered_actions(m.m)
POMDPTools.ordered_observations(m::CGCPProblem) = ordered_observations(m.m)
POMDPs.reward(m::CGCPProblem, s, a, sp) =  reward(m::CGCPProblem, s, a)

function POMDPs.reward(m::CGCPProblem, s, a)
    return m.initialized*reward(m.m, s, a) - dot(m.λ,cost(m.m,s,a))
end

function StateActionReward(m)
    return FunctionSARC(m)
end

PG_reward(m::CGCPProblem, s, a, sp) =  PG_reward(m, s, a)

function PG_reward(m::CGCPProblem,s,a)
    return [reward(m.m, s, a), ConstrainedPOMDPs.cost(m.m,s,a)...]
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
