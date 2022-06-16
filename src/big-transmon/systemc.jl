"""
system.jl - setting up the big transmon system and the dynamics with the continuous method
"""
#include the utils file
WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "utils.jl"))

using RobotDynamics
const RD = RobotDynamics
#number of transmon levels
const LEVELS = 10
#
# #Hilbert space dimensions
const HDIM = LEVELS
const HDIM_ISO = 2 * HDIM

const TRANSMON_FREQ = 0.0 #2pi * 4.96 GHz
const ANHARMONICITY = -2π * 0.143 #GHz
const CONTROL_COUNT = 2
const STATE_COUNT = 4

#some operators
# This is placing sqrt(1), ..., sqrt(TRANSMON_STATE_COUNT - 1) on the 1st diagonal
# # counted up from the true diagonal.
# const TRANSMON_ANNIHILATE = diagm(1 => map(sqrt, 1:LEVELS-1))
# # Taking the adjoint is as simple as adding an apostrophe.
# const TRANSMON_CREATE = TRANSMON_ANNIHILATE'
# const TRANSMON_NUMBER = TRANSMON_CREATE * TRANSMON_ANNIHILATE
# const TRANSMON_ID = I(LEVELS)
# const TRANSMON_QUAD = TRANSMON_NUMBER * (TRANSMON_NUMBER - TRANSMON_ID)

#astate, 4 states, controls, dcontrols

RD.@autodiff struct Transmon <: RobotDynamics.ContinuousDynamics
    ω_q::Float64 #qubit frequency in GHz
    α::Float64 #anharmonicity in GHz
    levels::Int
end
Transmon() = Transmon(TRANSMON_FREQ, ANHARMONICITY, LEVELS)
RobotDynamics.control_dim(::Transmon) = CONTROL_COUNT

Base.copy(model::Transmon) = Transmon(model.ω_q, model.α, model.levels)
#states + controls + dcontrols in astate
RobotDynamics.state_dim(model::Transmon) = STATE_COUNT*model.levels*2 + 2 * CONTROL_COUNT
nstates(::Transmon) = STATE_COUNT
qubit_dim_iso(model::Transmon) = model.levels*2
control_count(::Transmon) = CONTROL_COUNT

#some operators
function annihilate(model::Transmon)
    diagm(1 => map(sqrt, 1:model.levels - 1))
end

function create(model::Transmon)
    #apostrophe is adjoint in Julia
    (annihilate(model))'
end


function H0(model::Transmon)
    ω, α = model.ω_q, model.α
    return ω * create(model)*annihilate(model) +
    α/2 * create(model)*annihilate(model)*(create(model)*annihilate(model)- I(model.levels))
end

function H1R(model::Transmon)
    create(model) + annihilate(model)
end

function H1I(model::Transmon)
    1im * (create(model) - annihilate(model))
end

#methods to deal with the isomorphism
function getqstateinds(model::Transmon, i::Int)
    #2x for hdim_iso
    ir = (i-1)*2*model.levels
    return ir .+ (1:2*model.levels)
end

function getqstate_complex(model::Transmon, x, i::Int)
    ir = (i-1)*2*model.levels
    ic = ir + model.levels
    [x[ir+j] + x[ic+j]*1im for j in 1:model.levels]
end

function getqstate_iso(model::Transmon, x, i::Int)
    ir = (i-1)*2*model.levels
    x[ir + 1:ir + 2*model.levels]
end

function getcontrol(model::Transmon, astate, i::Int)
    astate[nstates(model)*model.levels*2 + i]
end

function getcontrols(model::Transmon, astate)
    astate[nstates(model)*model.levels*2 .+ (1:control_count(model))]
end

function getdcontrols(model::Transmon, astate)
    astate[nstates(model)*model.levels*2 + control_count(model) .+ (1:control_count(model))]
end

function getd2controls(model::Transmon, acontrol)
    acontrol[1:control_count(model)]
end

function control_idx(model::Transmon)
    return (1:control_count(model)) .+ nstates(model)*model.levels*2
end

function dcontrol_idx(model::Transmon)
    return nstates(model)*model.levels*2 + control_count(model) .+ (1:control_count(model))
end

function setqstate!(model::Transmon, x, ψ, i::Int)
    @assert length(ψ) == model.levels
    ir = (1:model.levels) .+ (i-1)*2*model.levels
    ic = ir .+ model.levels
    x[ir] .= real(ψ)
    x[ic] .= imag(ψ)
    return x
end


function setqstate_iso!(model::Transmon, x, ψ, i::Int)
    @assert length(ψ) == 2 * model.levels
    inds = (1:2*model.levels) .+ (i-1)*2*model.levels
    x[inds] .= ψ
end

function RobotDynamics.dynamics!(model::Transmon, xdot, x, u)
    negi_h = get_mat_iso(-1im * (H0(model) +
                        getcontrol(model, x, 1)* H1R(model)
                        + getcontrol(model, x, 2) * H1I(model)))
    state_shift = nstates(model)*model.levels*2

    dcontrols = getdcontrols(model,x)
    d2controls = getd2controls(model, u)
    #print(typeof(negi_h))
    #H_prop_iso = exp(negi_h * dt)

    for i = 1:nstates(model)
        ψi = getqstate_iso(model, x, i)
        ψdot = negi_h*ψi
        setqstate_iso!(model, xdot, ψdot, i)
    end

    xdot[control_idx(model)] = dcontrols
    xdot[dcontrol_idx(model)] = d2controls
end

function RobotDynamics.jacobian!(model::Transmon, J, xdot, x, u)
    J .= 0
    nquantum = nstates(model)*model.levels*2
    negi_h = get_mat_iso(-1im * (H0(model) +
                        getcontrol(model, x, 1)* H1R(model)
                        + getcontrol(model, x, 2) * H1I(model)))
    for i = 1:nstates(model)
        iψ = getqstateinds(model, i)
        ψi = getqstate_iso(model, x, i)
        J[iψ, iψ] .= negi_h
        J[iψ, nquantum + 1] = get_mat_iso(-1im*H1R(model)*getqstate(model,x, i))
        J[iψ, nquantum + 2] = get_mat_iso(-1im*H1I(model)*getqstate(model,x, i))
    end
    for i = 1:2
        J[nquantum + i, nquantum + 2 + i] = 1
        J[nquantum + i + 2, nquantum + 2 + i + 2] = 1
    end
    return nothing
end

RD.@autodiff struct DiscreteTransmon <: RobotDynamics.DiscreteDynamics
    continuous_model::Transmon
end
DiscreteTransmon(args...) = DiscreteTransmon(Transmon(args...))
@inline RD.state_dim(model::DiscreteTransmon) = RD.state_dim(model.continuous_model)
@inline RD.control_dim(model::DiscreteTransmon) = RD.control_dim(model.continuous_model)



function RobotDynamics.discrete_dynamics!(model::DiscreteTransmon, xn, x, u, t, dt)
    cmodel = model.continuous_model

    negi_h = get_mat_iso(-1im * (H0(cmodel) +
                        getcontrol(cmodel, x, 1)* H1R(cmodel)
                        + getcontrol(cmodel, x, 2) * H1I(cmodel)))
    #print(typeof(negi_h))
    H_prop_iso = exp(negi_h * dt)
    #h_prop = real2complex(h_prop_iso)
    controls_ = getcontrols(cmodel, x) + getdcontrols(cmodel, x) .* dt
    dcontrols_ = getdcontrols(cmodel, x) + getd2controls(u) .* dt

    for i = 1:nstates(cmodel)
        ψi_ = getqstate_iso(cmodel, x, i)
        ψi = H_prop_iso * ψi_
        setqstate_iso!(cmodel, xn, ψi, i)
    end
    #update with Euler
    xn[control_idx(cmodel)] = controls_
    xn[dcontrol_idx(cmodel)] = dcontrols_
    return nothing
end
