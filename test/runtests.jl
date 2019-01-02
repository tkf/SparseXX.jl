module Runtests
using SparseXX
using Test

@testset "$file" for file in [
        "test_vector.jl",
        ]
    include(file)
end

end  # module
