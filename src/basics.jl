@noinline notimplemented(f, args...) = throw(MethodError(f, args))

const DenseIndex = Union{UnitRange, Base.Slice{<:Base.OneTo}}
const SIMDArray{T} = Union{
    DenseArray{T},
    SubArray{T, <:Any, <:DenseArray{T},
             <:Tuple{<:DenseIndex, Vararg{Integer}}, true},
}

simdable(::T) where T = simdable(T)
simdable(::Type) = false
simdable(::Type{<: SIMD.ScalarTypes}) = true
simdable(::Type{<:SIMDArray{T}}) where T = simdable(T)
simdable(::Type{<:AdjOrTrans{<:Any, P}}) where P = simdable(P)

allsimdable() = true
allsimdable(x, args...) = simdable(x) && allsimdable(args...)

asdiag(A::Diagonal, n) = A.diag
asdiag(A::UniformScaling, n) = asdiag(A.λ, n)
asdiag(a::Number, n) = Fill(a, n, n)

isdiagtype(::Any) = false
isdiagtype(::Diagonal) = true
isdiagtype(::UniformScaling) = true
isdiagtype(::Number) = true
