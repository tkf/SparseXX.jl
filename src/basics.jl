butlast(x) = reverse(Base.tail(reverse(x)))

constructor_of(X) = constructor_of(typeof(X))
@generated constructor_of(::Type{T}) where T =
    getfield(parentmodule(T), nameof(T))

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
simdable(T::Type{<:SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = allsimdable(Tv, Ti)

allsimdable() = true
allsimdable(x, args...) = simdable(x) && allsimdable(args...)

asdiag(A::Diagonal, n) = A.diag
asdiag(A::UniformScaling, n) = asdiag(A.λ, n)
asdiag(a::Number, n) = Fill(a, n, n)

isdiagtype(::Any) = false
isdiagtype(::Diagonal) = true
isdiagtype(::UniformScaling) = true
isdiagtype(::Number) = true

@inline unsafe_column(A::AbstractMatrix, k) =
    SubArray(A, (Base.Slice(Base.OneTo(size(A, 1))), k))

rmul_or_fill!((Y, β)::Tuple{AbstractVecOrMat, Number}) = rmul_or_fill!(Y, β)
function rmul_or_fill!(Y::AbstractVecOrMat, β::Number)
    if β != 1
        β != 0 ? rmul!(Y, β) : fill!(Y, zero(eltype(Y)))
    end
    return
end

rmul_or_fill_many!() = nothing
function rmul_or_fill_many!(Yβ::Tuple, rest::Tuple...)
    rmul_or_fill!(Yβ)
    rmul_or_fill_many!(rest...)
end
