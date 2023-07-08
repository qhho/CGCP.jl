mutable struct CGCPProblem{S,A,O} <: POMDP{S, A, O}
    m::CPOMDP{S,A,O}
    λ::Vector{Float64}
    initialized::Bool
end

CGCPProblem(m::CPOMDP, λ, initialized=false) = CGCPProblem(m, λ, initialized)

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
POMDPs.isterminal(m::CGCPProblem, s) =  isterminal(m.m, s)

function POMDPs.reward(m::CGCPProblem, s, a)
    if !isterminal(m,s)
        return m.initialized*reward(m.m, s, a) - dot(m.λ,costs(m.m,s,a))
    else
        return 0.0
    end
end

PG_reward(m::CGCPProblem, s, a, sp) =  PG_reward(m, s, a)

PG_reward(m::CGCPProblem,s,a) = vcat(reward(m.m, s, a), costs(m.m,s,a))
