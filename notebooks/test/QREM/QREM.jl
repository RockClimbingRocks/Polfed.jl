
using SparseArrays, LinearAlgebra, Distributions, CUDA, Random
abstract type MODEL end


function construct_model(params::Dict)
    @unpack model_name = params


    if model_name == "qrem"
        return construct_qrem_model(params)
    elseif model_name == "qrem-rrg"
        return construct_qremrrg_model(params)
    else
        error("Model $model_name not implemented")
    end
    
end


struct QREM <: MODEL
    name::String
    L::Int
    hx::Float64
    spin::Float64
    local_dim::Int
    diags::Vector{Float64}
    hilbertspacedim::Int

    function QREM(name, L, hx, spin, diags)
        local_dim = Int(2spin+1)
        hilbertspacedim = local_dim^L
        new(name, L, hx, spin, local_dim, diags, hilbertspacedim)
    end
end


function get_basis_states(qrem::QREM)
    S = qrem.spin
    L = qrem.L
    d = Int(2S+1)
    D = d^L
    return collect(1:D)
end



function get_path_and_name(params::Dict, _::QREM)
    @unpack model_name, runname, L, hx, spin, avgs  = params

    params_name = copy(params)
    delete!(params_name, "runname")

    filepath = datadir("raw", runname, model_name, "spin_$(spin)", "L=$L", "hx=$hx")
    filename = savename(params_name) * ".h5"
    return filepath, filename
end

function get_path_and_name_collected(params::Dict, _::QREM)
    @unpack model_name, runname, L, hx, spin, avgs  = params
    
    params_name = copy(params)
    delete!(params_name, "runname")

    filepath = datadir("collected", runname, model_name, "spin_$(spin)", "L=$L")
    filename = savename(params_name) * ".h5"
    return filepath, filename
end







function construct_qrem_model(params::Dict)
    @unpack L, spin, model_name, hx, avgs = params

    D = Int(2*spin+1)^L
    μ, σ = 0, (L/2)^(0.5)
    d = Normal(μ, σ) 

    Random.seed!(1234+avgs) 

    random_energies = rand(d, D)  

    qrem = QREM(model_name, L, hx, spin, random_energies)
    return qrem 
end


include("cpu/cpu.jl")
include("gpu/gpu.jl")


function map_vec(qrem::QREM; a::Float64=1.0, b::Float64=0.0, pu::String="cpu")
    if pu == "cpu"
        return map_vec_cpu(qrem; a=a, b=b)
    elseif pu == "gpu"
        return map_vec_gpu(qrem; a=a, b=b)
    else
        throw(error("Unknown processing unit: $pu"))
    end
end


function clenshaw(qrem::QREM; a::Float64=1.0, b::Float64=0.0, pu::String="cpu")
    
    if pu == "cpu"
        return clenshaw_cpu(qrem; a=a, b=b)
    elseif pu == "gpu"
        return clenshaw_gpu(qrem; a=a, b=b)
    else
        throw(error("Unknown processing unit: $pu"))
    end
end


    # model:
    #   {
    #     name: "qrem",
    #     L: 10,
    #     hx: [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    #     spin: 0.5,
    #   },
