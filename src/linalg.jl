### mul!

for TA in [SparseXXMatrixCSC, AdjOrTrans{<:Any, <:SparseXXMatrixCSC}]
    @eval LinearAlgebra.mul!(C::StridedVecOrMat, A::$TA, B::StridedVecOrMat) =
        mul_api!(C, A, B, one(eltype(B)), zero(eltype(C)))
    @eval LinearAlgebra.mul!(C::StridedVecOrMat, A::$TA, B::StridedVecOrMat,
                             α::Number, β::Number) =
        mul_api!(C, A, B, α, β)
end

@inline function mul_api!(C, A, B, α, β)
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)
    if A isa AdjOrTrans
        if allsimdable(A, B)
            return mul_simd!(C, A, B, α, β)
        end
    elseif A isa SparseXXMatrixCSC
        if allsimdable(C, A)
            return mul_simd!(C, A, B, α, β)
        end
    end
    convertable_wo_copy(A) &&
        return mul!(C, convet(SparseMatrixCSC, A), B, α, β)
    return notimplemented(mul!, C, A, B, α, β)
end

"""
    mul_simd!(C, A::SparseXXMatrixCSC, B, α, β)

Compute ``C = C β + A α B``.
"""
@inline function mul_simd!(
        C, A::SparseXXMatrixCSC, B, α, β,
        ::Val{N} = Val(8),
        ::Val{align} = Val(false),
        ) where {N, align}
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end

    αdiag = asdiag(α, size(B, 1))
    nomask = Vec(ntuple(_ -> true, N))
    for k = 1:size(C, 2)
        Ck = unsafe_column(C, k)
        @inbounds for col = 1:A.n
            αxj = αdiag[col] * B[col, k]
            j = A.colptr[col]
            jmax = A.colptr[col + 1] - 1
            simd_end = jmax - N + 1
            while j <= simd_end
                idx = vload(Vec{N, eltype(rv)}, rv, j, Val{align})
                a = vload(Vec{N, eltype(nzv)}, nzv, j, Val{align})
                c = vgather(Ck, idx, nomask, Val{align})
                c = muladd(a, αxj, c)
                vscatter(c, Ck, idx, nomask, Val{align})
                j += N
            end
            while j <= jmax
                C[rv[j], k] += nzv[j] * αxj
                j += 1
            end
        end
    end
    return C
end

"""
    mul_simd!(C, A'::AdjOrTrans, B, α, β)

Compute ``C = C β + α A' B``.

!!! warning

    Note that the location of α is different from non-adjoint case.
    (TODO: This is totally ugly.  Fix it.)
"""
@inline function mul_simd!(C, adjA::AdjOrTrans, B, α, β)
    # eltype(A) is a SIMD.ScalarTypes hence is a Real; no need for
    # adjoint for each element
    A = adjA.parent
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    αdiag = asdiag(α, size(C, 1))
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            a = unsafe_column(A, col)
            b = unsafe_column(B, k)
            C[col, k] += αdiag[col] * dot_simd(a, b)
        end
    end
    return C
end


### fmul!

"""
    fmul!(Y, L, M, R)
    fmul!((Y, β), L, M, R)

Fused multiplication (and addition).

```math
Y = L M R + Y β
```

Following combinations are planned:

| ``L``        | ``M``        | ``R``        | done? |
|:---          |:---          |:---          |:---   |
| diagonal     | spmat'       | vec/mat      | [^1]  |
| spmat        | diagonal     | vec/mat      | [^1]  |

where

* diagonal: `Diagonal`, `UniformScaling`, or a `Number`
* spmat: `SparseXXMatrixCSC`
* spmat': `Adjoint` or `Transpose` of `SparseXXMatrixCSC`
* vec/mat: `AbstractVecOrMat`

[^1]: yes, for SIMD-able types

"""
fmul!(Y::AbstractVecOrMat, L, M, R) = fmul!((Y, false), L, M, R)

@inline function fmul!(Yβ::Tuple{AbstractVecOrMat, Number}, L, M, R)
    Y, β = Yβ
    if isdiagtype(L) && M isa AdjOrTrans && allsimdable(M, R)
        return mul_simd!(Y, M, R, L, β)
    elseif isdiagtype(M) && L isa SparseXXMatrixCSC && allsimdable(Y, L)
        return mul_simd!(Y, L, R, M, β)
    end
    notimplemented(fmul!, Yβ, L, M, R)
end


const M_M_VM = Tuple{AbstractMatrix,AbstractMatrix,AbstractVecOrMat}
const Diag_CSR_VM = Tuple{Diagonal,AdjOrTrans{<:Any,<:AbstractSparseMatrix},AbstractVecOrMat}
const CSC_Diag_VM = Tuple{AbstractSparseMatrix,Diagonal,AbstractVecOrMat}

const Diag_CSR = Tuple{Diagonal,AdjOrTrans{<:Any,<:AbstractSparseMatrix}}

isnzshared(A::AbstractSparseMatrix, B::AbstractSparseMatrix) =
    A.colptr === B.colptr && rowvals(A) === rowvals(B)
isnzshared(A::AdjOrTrans, B::AdjOrTrans) = isnzshared(parent(A), parent(B))

function spshared(S, nzval = similar(nonzeros(S)))
    m, n = size(S)
    colptr = S.colptr
    rowval = rowvals(S)
    return constructor_of(S)(m, n, colptr, rowval, nzval)
end

"""
    fmul_shared!((Y, β), (D1, S1', X1), ..., (Dn, Sn', Xn))
    fmul_shared!((Y, β), (D1, S1'), ..., (Dn, Sn'), X)
    fmul_shared!(((Y1, β1), ..., (Yn, βn)), (D1, S1', X1), ..., (Dn, Sn', Xn))
    fmul_shared!(((Y1, β1), ..., (Yn, βn)), (D1, S1'), ..., (Dn, Sn'), X)

Fused multiplications for sparse matrices with shared non-zero
structure.

```math
Y = Y β + D₁ S₁' X₁ + ... + Dₙ Sₙ' Xₙ

Y = Y β + (D₁ S₁' + ... + Dₙ Sₙ') X

Yᵢ = Yᵢ βᵢ + Dᵢ Sᵢ' Xᵢ

Yᵢ = Yᵢ βᵢ + Dᵢ Sᵢ' X
```

# Examples
```
julia> m = 3; n = 1; p = 0.1;

julia> begin
       using LinearAlgebra, SparseArrays, SparseXX
       using Random
       S1 = SparseXXMatrixCSC(sprandn(m, m, p))
       S2 = spshared(S1)
       randn!(nonzeros(S2))
       X1 = randn(m, n)
       X2 = randn(m, n)
       D1 = Diagonal(randn(m))
       D2 = Diagonal(randn(m))
       Y1 = zero(X1)
       Y2 = zero(X1)
       end;

julia> fmul_shared!(Y1, (D1, S1', X1), (D2, S2', X2)) === Y1
true

julia> fmul_shared!((Y1, Y2), (D1, S1', X1), (D2, S2', X2)) === (Y1, Y2)
true

julia> fmul_shared!(Y1, (D1, S1'), (D2, S2'), X1) === Y1
true

julia> fmul_shared!((Y1, Y2), (D1, S1'), (D2, S2'), X1) === (Y1, Y2)
true
```
"""
@inline function fmul_shared!(Yβ_, rhs_...)
    check_fmul_shared_args(Yβ_, rhs_)
    Yβ = canonicalize_Yβ(Yβ_)
    rhs = canonicalize_rhs(rhs_)
    if is_shared_simd3(rhs)
        return fmul_shared_simd3!(Yβ, rhs...)
    elseif is_shared_simd2(rhs)
        return _fmul_shared_simd!(Val(4), Yβ, butlast(rhs), rhs[end])
    end
    notimplemented(fmul_shared!, Yβ, rhs...)
end
# It would be nice to have some computation graph exectuor on top of
# fmul*!, but it can be done later.

@inline function check_fmul_shared_args(Yβ, rhs)
    if length(rhs) == 0
        throw(ArgumentError("fmul_shared! needs one or more `rhs` arguments"))
    end

    # Fan-out case
    if Yβ isa Union{Tuple{Vararg{AbstractMatrix}},
                    Tuple{Vararg{Tuple{AbstractMatrix,Number}}}}
        if rhs[end] isa AbstractVecOrMat
            if length(Yβ) != length(rhs) - 1
                throw(ArgumentError("""
Detected call signature:
    fmul_shared!((Yβ1, ..., Yβm), (D1, S1'), ..., (Dn, Sn'), X)
with `m = $(length(Yβ))` and `n = $(length(rhs) - 1)`.  Note that `m` and `n` must match."""))
            end
        else
            if length(Yβ) != length(rhs)
                throw(ArgumentError("""
Detected call signature:
    fmul_shared!((Yβ1, ..., Yβm), (D1, S1', X1), ..., (Dn, Sn', Xn))
with `m = $(length(Yβ))` and `n = $(length(rhs))`.  Note that `m` and `n` must match."""))
            end
        end
    end

    if Yβ isa Tuple{Vararg{AbstractMatrix}}
        n1, n4 = size(Yβ[1])
        if !all(let dim = (n1, n4)
                    Y -> size(Y) == dim
                end,
                Yβ)
            throw(ArgumentError("""
Matrices `Y1`, ..., `Yn` passed to `fmul_shared!((Y1, ..., Yn), ...)` do not
have uniform `size`."""))
        end
    elseif Yβ isa Tuple{Vararg{Tuple{AbstractMatrix,Number}}}
        n1, n4 = size(Yβ[1][1])
        if !all(let dim = (n1, n4)
                    ((Y, _),) -> size(Y) == dim
                end,
                Yβ)
            throw(ArgumentError("""
Matrices `Y1`, ..., `Yn` passed to `fmul_shared!(((Y1, β1), ..., (Yn, βn)), ...)`
do not have uniform `size`."""))
        end
    elseif Yβ isa AbstractMatrix
        n1, n4 = size(Yβ)
    elseif Yβ isa Tuple{AbstractMatrix,Number}
        n1, n4 = size(Yβ[1])
    else
        throw(ArgumentError("""
        Unsupported type for first argument of `fmul_shared!`:
            $(typeof(Yβ))
        """))
    end

    if rhs[end] isa AbstractVecOrMat
        terms = butlast(rhs)
        if length(terms) == 0
            throw(ArgumentError("""
Invalid call signature:
    fmul_shared!(Yβ, X)
See `?fmul_shared!` for the list of supported call signatures.
"""))
        elseif !(terms isa Tuple{Vararg{Tuple{DiagonalLike,AbstractMatrix}}})
            throw(ArgumentError("""
Detected call signature:
    fmul_shared!(Yβ, (D1, S1'), ..., (Dn, Sn'), X)
However, one of `(Di, Si')` is not of supported type.  Each `Di` must be of
a diagonal-like type (i.e.,`$DiagonalLike`)
and `Si'` must be a matrix.
"""))
        end
    else
        terms = rhs
        if !(terms isa Tuple{Vararg{Tuple{DiagonalLike,AbstractMatrix,AbstractVecOrMat}}})
            throw(ArgumentError("""
Detected call signature:
    fmul_shared!(Yβ, (D1, S1', X1), ..., (Dn, Sn', X2))
However, one of `(Di, Si', Xi)` is not of supported type.  Each `Di` must be of
a diagonal-like type (i.e.,`$DiagonalLike`),
`Si'` must be and a matrix, and `Xi` must be a matrix or a vector.
"""))
        end
    end

    S1 = rhs[1][2] :: AbstractMatrix
    n2, n3 = size(S1)
    if rhs[end] isa AbstractVecOrMat
        if matsize(rhs[end]) != (n3, n4)
            throw(ArgumentError("""
Size of argument `X` passed to `fmul_shared!` is inconsistent with `Y`.
"""))
        end
    else
        if !all(let dim = (n3, n4)
                    ((_, _, X),) -> matsize(X) == dim
                end,
                terms)
            throw(ArgumentError("""
Sizes of arguments `X` passed to `fmul_shared!` are inconsistent.
"""))
        end
    end

    if !all(let dim = (n2, n3)
                ((_, S),) -> size(S) == dim
            end,
            terms)
        throw(ArgumentError("""
Sizes of arguments `S` passed to `fmul_shared!` are inconsistent.
"""))
    elseif !all(let dim = (n1, n2)
                    ((D,),) -> !(D isa Diagonal) || size(D) == dim
                end,
                terms)
        throw(ArgumentError("""
Sizes of arguments `D` passed to `fmul_shared!` are inconsistent.
"""))
    end

    return
end

@inline canonicalize_rhs(rhs) = map(canonicalize_term, rhs)

@inline canonicalize_term(term) = term
@inline function canonicalize_term(DSX::Tuple{DiagonalLike,Any,Vararg})
    D, S = DSX
    return (Diagonal(asdiag(D, size(S, 1))), Base.tail(DSX)...)
end

@inline canonicalize_Yβ(Yβ::Tuple{AbstractMatrix,Number}) = Yβ
@inline canonicalize_Yβ(Y::AbstractMatrix) = (Y, false)

@inline canonicalize_Yβ(Yβ::Tuple{Vararg{Tuple{AbstractMatrix,Number}}}) = Yβ
@inline canonicalize_Yβ(Ys::Tuple{Vararg{AbstractMatrix}}) =
    map(canonicalize_Yβ, Ys)

@inline function is_shared_simd3(triplets)
    t1 = triplets[1]
    rest = Base.tail(triplets)
    return triplets isa Tuple{Vararg{Diag_CSR_VM}} &&
        all(((D, S, X),) -> allsimdable(S, X), triplets) &
        all(((_, S, _),) -> isnzshared(t1[2], S), rest)
end

@inline function is_shared_simd2(pairs_and_vm)
    pairs = butlast(pairs_and_vm)
    t1 = pairs_and_vm[1]
    middle = Base.tail(pairs)
    vm = pairs_and_vm[end]
    return simdable(vm) &&
        pairs isa Tuple{Vararg{Diag_CSR}} &&
        all(((_, S),) -> simdable(S), pairs) &&
        all(((_, S),) -> isnzshared(t1[2], S), middle)
end

@inline function preprocess_Yβ(Yβ::Tuple{AbstractMatrix,Number})
    rmul_or_fill!(Yβ)
    return Yβ[1]
end

@inline function preprocess_Yβ(Yβs::Tuple{Vararg{Tuple{AbstractMatrix,Number}}})
    rmul_or_fill_many!(Yβs...)
    return map(first, Yβs)
end

@inline fmul_shared_simd3!(Yβ, triplets...) =
    _fmul_shared_simd!(Val(4), Yβ, triplets)

# Using `triplets::Tuple{Vararg{Diag_CSR_VM}}` instead of
# `triplets::Diag_CSR_VM...` seems to be important for Julia to
# optimize this function.
@inline function _fmul_shared_simd!(
        ::Val{N},
        Yβ, triplets::Tuple{Vararg{Diag_CSR_VM}}
        ) where {N}

    Y = preprocess_Yβ(Yβ)
    Y :: Union{AbstractMatrix,Tuple{Vararg{AbstractMatrix}}}

    lane = VecRange{N}(0)

    S1 = parent(triplets[1][2])
    nzind = rowvals(S1)
    nzvs = map(((_, S, _),) -> nonzeros(parent(S)), triplets)
    diags = map(((D, _, _),) -> D.diag, triplets)

    for k = 1:size(triplets[1][3], 2)
        Xs = let k = k
            map(DSX -> unsafe_column(DSX[3], k), triplets)
        end
        @inbounds for col = 1:S1.n
            vaccs = map(triplets) do (_, S, _)
                zero(Vec{N, eltype(S)})  # TODO: promote
            end

            nzr = nzrange(S1, col)
            simd_end = last(nzr) - N + 1
            j = first(nzr)
            while j <= simd_end
                #=
                let lane = lane,
                    idx = nzind[lane + j] #=,
                    j = j =#
                    vaccs = map(vaccs, nzvs, Xs) do vacc, nzv, X
                        @inbounds muladd(nzv[lane + j], X[idx], vacc)
                    end
                end
                =#
                vaccs = compute_vaccs(vaccs, nzvs, Xs, nzind[lane + j], lane + j)
                j += N
            end

            accs = map(sum, vaccs)
            while j <= last(nzr)
                accs = let j = j, idx = nzind[j]
                    map(accs, nzvs, Xs) do acc, nzv, X
                        @inbounds muladd(nzv[j], X[idx], acc)
                    end
                end
                j += 1
            end

            update_Y!(Y, diags, accs, col, k)
        end
    end
    return Y
end

@inline function update_Y!(Y::AbstractMatrix, diags, accs, col, k)
    prods = map(diags, accs) do diag, acc
        @inbounds diag[col] * acc
    end
    @inbounds Y[col, k] += +(prods...)
    return
end

@inline function update_Y!(Ys::Tuple{Vararg{AbstractMatrix}},
                           diags, accs, col, k)
    map(Ys, diags, accs) do Y, diag, acc
        @inbounds Y[col, k] += diag[col] * acc
        return
    end
    return
end

# This is a workaround for the possible bug in `let`:
# https://github.com/JuliaLang/julia/issues/30951.  This fixes the
# test _and_ makes the inference work.
compute_vaccs(::Tuple{}, ::Tuple{}, ::Tuple{}, _, _) = ()
@inline compute_vaccs(vaccs, nzvs, Xs, idx, vj) =
    ((@inbounds muladd(nzvs[1][vj], Xs[1][idx], vaccs[1])),
     compute_vaccs(Base.tail(vaccs),
                   Base.tail(nzvs),
                   Base.tail(Xs),
                   idx, vj)...)

@inline function _fmul_shared_simd!(
        ::Val{N},
        Yβ, pairs::Tuple{Vararg{Diag_CSR}}, X,
        ) where {N}

    Y = preprocess_Yβ(Yβ)
    Y :: Union{AbstractMatrix,Tuple{Vararg{AbstractMatrix}}}

    lane = VecRange{N}(0)

    S1 = parent(pairs[1][2])
    nzind = rowvals(S1)
    nzvs = map(((_, S),) -> nonzeros(parent(S)), pairs)
    diags = map(((D, _),) -> D.diag, pairs)

    for k = 1:size(X, 2)
        Xk = unsafe_column(X, k)
        @inbounds for col = 1:S1.n
            vaccs = map(pairs) do (_, S)
                zero(Vec{N, eltype(S)})  # TODO: promote
            end

            nzr = nzrange(S1, col)
            simd_end = last(nzr) - N + 1
            j = first(nzr)
            while j <= simd_end
                #=
                vaccs = let lane = lane,
                    Xjk = Xk[nzind[lane + j]],
                    j = j
                    map(vaccs, nzvs) do vacc, nzv
                        @inbounds muladd(nzv[lane + j], Xjk, vacc)
                    end
                end
                =#
                vaccs = compute_vaccs(vaccs, nzvs, Xk[nzind[lane + j]], lane + j)
                j += N
            end

            accs = map(sum, vaccs)
            while j <= last(nzr)
                accs = let j = j, Xjk = Xk[nzind[j]]
                    map(accs, nzvs) do acc, nzv
                        @inbounds muladd(nzv[j], Xjk, acc)
                    end
                end
                j += 1
            end

            update_Y!(Y, diags, accs, col, k)
        end
    end
    return Y
end

compute_vaccs(::Tuple{}, ::Tuple{}, _, _) = ()
@inline compute_vaccs(vaccs, nzvs, Xjk, vj) =
    ((@inbounds muladd(nzvs[1][vj], Xjk, vaccs[1])),
     compute_vaccs(Base.tail(vaccs),
                   Base.tail(nzvs),
                   Xjk, vj)...)
