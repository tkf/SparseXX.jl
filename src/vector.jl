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
                     Vv<:AbstractVector{Ti}, Vi<:AbstractVector{Tv}}
        n >= 0 ||
            throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(ArgumentError("index and value vectors must be the same length"))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseXXVector{Tv,Ti}(
        n::Integer,
        nzind::AbstractVector{Ti},
        nzval::AbstractVector{Tv}) where {Tv, Ti} =
    SparseXXVector{eltype(nzval), eltype(nzind),
                   typeof(nzval), typeof(nzind)}(n, nzind, nzval)

SparseArrays.SparseVector(v::SparseXXVector) =
    SparseVector(v.n, Vector(v.nzind), Vector(v.nzval))
