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
    fmul_shared!((Y, β), (S1, D1, X1), ..., (Sn, Dn, Xn))

Fused multiplications for sparse matrices with shared non-zero
structure.

```math
Y = Y β + D₁ S₁' X₁ + ... + Dₙ Sₙ' Xₙ

Y = Y β + S₁ D₁ X₁ + ... + Sₙ Dₙ Xₙ
```

(ATM, only the first form with SIMD-compatible scalar types is defined.)

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
       Y = zero(X1)
       end;

julia> fmul_shared!(Y, (D1, S1', X1), (D2, S2', X2))
```
"""
fmul_shared!(Y::AbstractVecOrMat, triplets...) =
    fmul_shared!((Y, false), triplets...)

@inline function fmul_shared!(Yβ::Tuple{AbstractVecOrMat, Number},
                              triplets::M_M_VM...)
    if is_shared_csr_simd(triplets)
        return fmul_shared_simd!(Yβ, triplets...)
    end
    notimplemented(fmul_shared!, Yβ, triplets...)
end
# It would be nice to have some computation graph exectuor on top of
# fmul*!, but it can be done later.

@inline function is_shared_csr_simd(triplets)
    t1 = triplets[1]
    rest = triplets[2:end]
    return all(isa.(triplets, Diag_CSR_VM)) &&
        all(((D, S, X),) -> allsimdable(S, X), triplets) &
        all(((_, S, _),) -> isnzshared(t1[2], S), rest)
end

@inline function fmul_shared_simd!(
        Yβ, triplets::Diag_CSR_VM...;
        simdwidth::Val{N} = Val(4)
        ) where {N}
    Y, β = Yβ
    if β != 1
        β != 0 ? rmul!(Y, β) : fill!(Y, zero(eltype(Y)))
    end

    lane = VecRange{N}(0)

    S1 = parent(triplets[1][2])
    nzind = rowvals(S1)
    nzvs = map(((_, S, _),) -> nonzeros(parent(S)), triplets)
    diags = map(((D, _, _),) -> D.diag, triplets)

    for k = 1:size(Y, 2)
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

            prods = let col = col
                map(diags, accs) do diag, acc
                    @inbounds diag[col] * acc
                end
            end
            Y[col, k] += +(prods...)
        end
    end
    return Y
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
