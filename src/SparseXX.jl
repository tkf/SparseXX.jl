module SparseXX

export SparseXXVector, SparseXXMatrixCSC

using LinearAlgebra
using LinearAlgebra: AdjOrTrans
using SparseArrays

using SIMD

include("basics.jl")
include("matrix.jl")
include("vector.jl")
include("linalg.jl")

end # module
