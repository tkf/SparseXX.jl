"""
    SparseXXMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
"""
struct SparseXXMatrixCSC{Tv, Ti<:Integer,
                         Tc<:AbstractVector{Ti},
                         Tr<:AbstractVector{Ti},
                         Tz<:AbstractVector{Tv}} <: AbstractSparseMatrixCSC{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    colptr::Tc              # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::Tr              # Row indices of stored values
    nzval::Tz               # Stored values, typically nonzeros

    function SparseXXMatrixCSC{Tv,Ti,Tc,Tr,Tz}(
            m::Integer, n::Integer,
            colptr::Tc, rowval::Tr, nzval::Tz) where {
                Tv, Ti<:Integer,
                Tc<:AbstractVector{Ti},
                Tr<:AbstractVector{Ti},
                Tz<:AbstractVector{Tv}}
        @noinline throwsz(str, lbl, k) =
            throw(ArgumentError("number of $str ($lbl) must be ≥ 0, got $k"))
        m < 0 && throwsz("rows", 'm', m)
        n < 0 && throwsz("columns", 'n', n)
        new(Int(m), Int(n), colptr, rowval, nzval)
    end
end

SparseXXMatrixCSC(m::Integer, n::Integer, colptr::AbstractVector,
                  rowval::AbstractVector, nzval::AbstractVector) =
    SparseXXMatrixCSC{eltype(nzval), eltype(colptr),
                      typeof(colptr), typeof(rowval),
                      typeof(nzval)}(m, n, colptr, rowval, nzval)

ColPtrType(::Type{<:SparseXXMatrixCSC{<:Any, <:Any, Tc}}) where Tc = Tc
RowValType(::Type{<:SparseXXMatrixCSC{<:Any, <:Any, <:Any, Tr}}) where Tr = Tr
NZValType(::Type{<:SparseXXMatrixCSC{<:Any, <:Any, <:Any, <:Any, Tz}}) where Tz = Tz

simdable(T::Type{<:SparseXXMatrixCSC}) =
    allsimdable(ColPtrType(T), RowValType(T), NZValType(T))

SparseXXMatrixCSC(S::SparseMatrixCSC) =
    SparseXXMatrixCSC(S.m, S.n, S.colptr, S.rowval, S.nzval)

SparseArrays.SparseMatrixCSC(S::SparseXXMatrixCSC) =
    SparseMatrixCSC(S.m, S.n,
                    convert(Vector, S.colptr),
                    convert(Vector, S.rowval),
                    convert(Vector, S.nzval))

convertable_wo_copy(::T) where {T <: SparseXXMatrixCSC} =
    ColPtrType(T) isa Vector &&
    RowValType(T) isa Vector &&
    NZValType(T) isa Vector

@inline function unsafe_column(S::AbstractSparseMatrix, col)
    idx = ((@inbounds S.colptr[col]:(S.colptr[col + 1] - 1)),)
    return SparseXXVector(S.n, SubArray(S.rowval, idx), SubArray(S.nzval, idx))
end

### Basic properties

Base.size(S::SparseXXMatrixCSC) = (S.m, S.n)

SparseArrays.nnz(S::SparseXXMatrixCSC) = Int(S.colptr[S.n + 1] - 1)
SparseArrays.nnz(S::Base.ReshapedArray{T,1,<:SparseXXMatrixCSC}) where T =
    nnz(parent(S))

SparseArrays.nonzeros(S::SparseXXMatrixCSC) = S.nzval
SparseArrays.rowvals(S::SparseXXMatrixCSC) = S.rowval
Base.@propagate_inbounds SparseArrays.nzrange(S::SparseXXMatrixCSC, col::Integer) =
    S.colptr[col]:(S.colptr[col+1]-1)


function Base.getindex(S::SparseXXMatrixCSC, i::Integer, j::Integer)
    nzr = nzrange(S, j)
    nzv = view(nonzeros(S), nzr)
    nzi = view(rowvals(S), nzr)
    k = searchsortedfirst(nzi, i)
    if k > length(nzi) || nzi[k] != i
        return zero(eltype(nzv))
    end
    return nzv[k]
end
