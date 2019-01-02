"""
    SparseXXVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}

Vector type for storing sparse vectors.
"""
struct SparseXXVector{
        Tv,
        Ti <: Integer,
        Vv <: AbstractVector{Tv},
        Vi <: AbstractVector{Ti},
        } <: AbstractSparseVector{Tv,Ti}
    n::Int              # Length of the sparse vector
    nzind::Vi           # Indices of stored values
    nzval::Vv           # Stored values, typically nonzeros

    function SparseXXVector{Tv,Ti,Vv,Vi}(
            n::Integer, nzind::Vi, nzval::Vv
            ) where {Tv, Ti<:Integer,
                     Vv<:AbstractVector{Tv}, Vi<:AbstractVector{Ti}}
        n >= 0 ||
            throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(ArgumentError("index and value vectors must be the same length"))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseXXVector(
        n::Integer,
        nzind::AbstractVector,
        nzval::AbstractVector) =
    SparseXXVector{eltype(nzval), eltype(nzind),
                   typeof(nzval), typeof(nzind)}(n, nzind, nzval)

SparseXXVector(v::SparseVector) = SparseXXVector(v.n, v.nzind, v.nzval)

SparseArrays.SparseVector(v::SparseXXVector) =
    SparseVector(v.n, convert(Vector, v.nzind), convert(Vector, v.nzval))

convertable_wo_copy(::Any) = false
convertable_wo_copy(::SparseXXVector{<:Any, <:Any, <:Vector, <:Vector}) = true

### Basic properties

Base.length(x::SparseXXVector) = x.n
Base.size(x::SparseXXVector) = (x.n,)
SparseArrays.nnz(x::SparseXXVector) = length(x.nzval)
Base.count(f, x::SparseXXVector) =
    count(f, x.nzval) + f(zero(eltype(x))) * (length(x) - nnz(x))

SparseArrays.nonzeros(x::SparseXXVector) = x.nzval
SparseArrays.nonzeroinds(x::SparseXXVector) = x.nzind

### Dot product

LinearAlgebra.dot(x::AbstractVector, y::SparseXXVector) = dot_dispatch(x, y)
LinearAlgebra.dot(x::SparseXXVector, y::AbstractVector) = dot_dispatch(x, y)

@inline function dot_dispatch(x, y)
    @assert length(x) == length(y)
    if eltype(x) <: SIMD.ScalarTypes && eltype(y) <: SIMD.ScalarTypes
        if x isa SparseXXVector && y isa DenseVector
            return dot_simd(x, y)
        elseif x isa DenseVector && y isa SparseXXVector
            return dot_simd(y, x)
        end
    end
    convertable_wo_copy(x) && return dot(convert(SparseVector, x), y)
    convertable_wo_copy(y) && return dot(x, convert(SparseVector, y))
    return notimplemented(dot, x, y)
end

@inline function dot_simd(sv::SparseXXVector, ys::SIMDArray,
                          ::Val{align} = Val(false),
                          ::Val{N} = Val(4),
                          ) where {N, align}
    Ti = eltype(sv.nzind)
    Tv = eltype(sv.nzval)
    T = promote_type(eltype(sv.nzval), eltype(ys))
    isempty(sv.nzval) && return zero(T)

    nz_size = length(sv.nzval)
    nomask = Vec(ntuple(_ -> true, N))
    vacc = zero(Vec{N, T})
    simd_end = nz_size - N + 1
    j = 1
    @inbounds while j <= simd_end
        idx = vload(Vec{N, Ti}, sv.nzind, j, Val{align})
        bs = vgather(ys, idx, nomask, Val{align})
        as = vload(Vec{N, Tv}, sv.nzval, j, Val{align})
        vacc = muladd(as, bs, vacc)
        j += N
    end

    acc = sum(vacc)
    @inbounds while j <= nz_size
        acc = muladd(sv.nzval[j], ys[sv.nzind[j]], acc)
        j += 1
    end
    return acc
end
