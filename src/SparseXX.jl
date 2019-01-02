module SparseXX

export SparseXXVector, SparseXXMatrixCSC

using LinearAlgebra
using SparseArrays

using SIMD

include("basics.jl")
include("matrix.jl")
include("vector.jl")

end # module
