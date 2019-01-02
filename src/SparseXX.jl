module SparseXX

export SparseXXVector, SparseXXMatrixCSC, fmul!

using LinearAlgebra
using LinearAlgebra: AdjOrTrans
using SparseArrays

using FillArrays
using SIMD

include("basics.jl")
include("matrix.jl")
include("vector.jl")
include("linalg.jl")

end # module
