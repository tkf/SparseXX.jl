"""
    SparseXXMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
"""
struct SparseXXMatrixCSC{Tv, Ti<:Integer,
                         Tc<:AbstractVector{Ti},
                         Tr<:AbstractVector{Ti},
                         Tz<:AbstractVector{Tv}} <: AbstractSparseMatrix{Tv,Ti}
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
