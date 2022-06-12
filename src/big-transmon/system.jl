"""
system.jl - setting up the big transmon system and the dynamics
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

RD.@autodiff struct BigTransmon <: RobotDynamics.ContinuousDynamics
    ω_q::Float64 #qubit frequency in GHz
    α::Float64 #anharmonicity in GHz
    levels::Int
end
BigTransmon() = BigTransmon(TRANSMON_FREQ, ANHARMONICITY, LEVELS)
RobotDynamics.control_dim(::BigTransmon) = CONTROL_COUNT
#states + controls + dcontrols in astate
RobotDynamics.state_dim(::BigTransmon) = STATE_COUNT*HDIM_ISO + 2 * CONTROL_COUNT
nstates(::BigTransmon) = STATE_COUNT
qubit_dim_iso(::BigTransmon) = HDIM_ISO

#some operators
function annihilate(model::BigTransmon)
    diagm(1 => map(sqrt, 1:model.levels - 1))
end

function create(model::BigTransmon)
    #apostrophe is adjoint in Julia
    (annihilate(model))'
end


function H0(model::BigTransmon)
    ω, α = model.ω_q, model.α
    ω * create(model)*annihilate(model) +
    α/2 * create(model)*annihilate(model)(create(model)*annihilate(model)- I(model.levels))
end

function H1R(model::BigTransmon)
    create(model) + annihilate(model)
end

function H1I(model::BigTransmon)
    1im * (create(model) - annihilate(model))
end

#methods to deal with the isomorphism
function getqstateinds(model::BigTransmon, i::Int)
    #2x for hdim_iso
    ir = (i-1)*2*model.levels
    return ir .+ (1:2*model.levels)
end

function getqstate(model::BigTransmon, x, i::Int)
    ir = (i-1)*2*model.levels
    ic = ir + model.levels
    [x[ir+j] + x[ic+j]*1im for j in 1:model.levels]
end

function getcontrol(model::BigTransmon, astate, i::Int)
    astate[nstates(model)*model.levels*2 + i]
end

function getdcontrol(model::BigTransmon, astate, i::Int)
    astate[nstates(model)*model.levels*2 + CONTROL_COUNT + i]
end

function setqstate!(model::BigTransmon, x, ψ, i::Int)
    @assert length(ψ) == model.levels
    ir = (1:model.levels) .+ (i-1)*2*model.levels
    ic = ir .+ model.levels
    x[ir] .= real(ψ)
    x[ic] .= imag(ψ)
    return x
end

function RD.discrete_dynamics(model::BigTransmon, astate::AbstractVector, acontrol::AbstractVector, time::Real, dt::Real)
    negi_h = complex2real(-1im * (H0(model) +
                        getcontrol(model, astate, 1)* H1R(model)
                        + getcontrol(model, astate, 2) * H1I(model)))
    h_prop_iso = exp(negi_h * dt)
    h_prop = real2complex(h_prop_iso)

end
