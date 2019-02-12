module SparseXX

export SparseXXVector, SparseXXMatrixCSC, fmul!, fmul_shared!, spshared,
    asfmulable

using Base: tail
using Base.Broadcast: broadcasted

using LinearAlgebra
using LinearAlgebra: AdjOrTrans
using SparseArrays

using FillArrays
using SIMD

include("basics.jl")
include("matrix.jl")
include("vector.jl")
include("linalg.jl")
include("chained.jl")
include("interop_blockarrays.jl")

end # module
