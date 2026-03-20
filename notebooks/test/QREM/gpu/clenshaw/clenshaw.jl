
include("spin0.5/reocurence_relation.jl")
include("spin0.5/final_sum.jl")



function clenshaw_gpu(qrem::QREM; a::Real=1., b::Real=0.)
    # diags = get_diags(qrem, a, b)
    diags_cu = (CuVector(qrem.diags) .- b) ./ a  

    #just to prepare data to put it into function 
    spin = qrem.spin
    Leff = qrem.L - 1
    hx = 0.5*qrem.hx / a

    # Running params
    max_threads_per_block = 1024
    basis_length = Int(2spin+1)^qrem.L


    crr, cfs = nothing, nothing


    if spin≈0.5
        crr = @inline (b1::CuVecOrMat, b2::CuVecOrMat, b3::CuVecOrMat, c::Real, X::CuVecOrMat) -> begin

            if X isa AbstractVector
                threads_per_block = max_threads_per_block   
                blocks_per_grid = cld(basis_length, threads_per_block)
                @cuda blocks=blocks_per_grid threads=threads_per_block clenshaw_reocurence_relation_vec_kernel_spin_onehalf(b1, b2, b3, c, X, hx, diags_cu, Leff, basis_length)

            elseif X isa AbstractMatrix
                block_width = size(X, 2)
                num_matrix_elem = basis_length*block_width
                threads_per_block = max_threads_per_block

                blocks_per_grid = cld(num_matrix_elem, threads_per_block)
                @cuda blocks=blocks_per_grid threads=threads_per_block clenshaw_reocurence_relation_mat_kernel_1d_spin_onehalf(b1, b2, b3, c, X, hx, diags_cu, Leff, basis_length, num_matrix_elem)
            else
                throw(error("Something is not OK!!! Either the block is to width or somthing else (type)"))
            end
        end

        cfs = @inline (b1::CuVecOrMat, b2::CuVecOrMat, c::Real, Y::CuVecOrMat, X::CuVecOrMat) -> begin
            @assert size(Y)==size(X) "Size of Y and X are not the same! X is of size $(size(X)) and Y is of size $(size(Y))"

            if X isa AbstractVector
                threads_per_block = max_threads_per_block   
                blocks_per_grid = cld(basis_length, threads_per_block)
                @cuda blocks=blocks_per_grid threads=threads_per_block clenshaw_finalsum_vec_kernel_spin_onehalf(b1, b2, c, Y, X, hx, diags_cu, Leff, basis_length)

            elseif X isa AbstractMatrix
                block_width = size(X, 2)
                num_matrix_elem = basis_length*block_width
                threads_per_block = max_threads_per_block

                blocks_per_grid = cld(num_matrix_elem, threads_per_block)
                @cuda blocks=blocks_per_grid threads=threads_per_block clenshaw_finalsum_mat_kernel_1d_spin_onehalf(b1, b2, c, Y, X, hx, diags_cu, Leff, basis_length, num_matrix_elem)
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


    
    return crr, cfs    
end





