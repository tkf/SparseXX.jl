module Runtests
using SparseXX
using Test

@testset "$file" for file in [
        "test_vector.jl",
        "test_matrix.jl",
        ]
    include(file)
end

end  # module
