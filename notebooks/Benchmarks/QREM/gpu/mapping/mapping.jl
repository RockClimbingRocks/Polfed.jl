
include("spin_one_half/vector_mapping.jl")
include("spin_one_half/matrix_mapping.jl")


function map_vec_gpu(qrem::QREM; a::Real=1., b::Real=0.)

    # diags = get_diags(qrem, a, b)
    diags_cu = (CuVector(qrem.diags) .- b) ./ a  

    #just to prepare data to put it into function 
    spin = qrem.spin
    L = qrem.L
    Leff = L - 1
    hx =0.5 * qrem.hx / a

    # Running params
    max_threads_per_block = 1024
    basis_length = Int(2spin+1)^L


    if spin≈0.5
        return @inline (Y::CuVecOrMat, X::CuVecOrMat) -> begin
            @assert size(Y)==size(X) "Size of Y and X are not the same! X is of size $(size(X)) and Y is of size $(size(Y))"

            if X isa AbstractVector
                threads_per_block = max_threads_per_block   
                blocks_per_grid = cld(basis_length, threads_per_block)
                
                @cuda blocks=blocks_per_grid threads=threads_per_block qrem_map_kernel_spin_half(Y, X, Leff, diags_cu, hx, basis_length)

            elseif X isa AbstractMatrix
                block_width = size(X, 2)
                num_matrix_elem = basis_length*block_width
                threads_per_block = max_threads_per_block

                blocks_per_grid = cld(num_matrix_elem, threads_per_block)
                @cuda blocks=blocks_per_grid threads=threads_per_block qrem_map_mat_kernel_1d_spin_half(Y, X, Leff, diags_cu, hx, basis_length, num_matrix_elem)
            else
                throw(error("Something is not OK!!! Either the block is to width or somthing else (type)"))
            end
        end

    end


    if spin ≈ 1.
        throw(error("not implemented yet!!!"))
    end

    if spin ≈ 1.5
        throw(error("not implemented yet!!!"))
    end
end



