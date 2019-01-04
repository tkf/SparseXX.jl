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

@generated function fmul_shared_simd!(Yβ, triplets::Diag_CSR_VM...)
    N = 4
    align = Val{false}
    nt = length(triplets)

    init_vaccs = quote end
    simd_body = quote
        idx = vload(Vec{$N, Ti}, nzind, j, $align)
    end
    reduce_vaccs = quote end
    scalar_body = quote end
    mul_diag_exprs = []
    for i in 1:nt
        TS = triplets[i].types[2]  # sparse matrix types
        Tv = eltype(TS)
        Ta = Tv  # TODO: promote
        vacc = Symbol("vacc", i)
        acc = Symbol("acc", i)
        init_vaccs = quote
            $init_vaccs
            $vacc = zero(Vec{$N, $Ta})
        end
        vx = Symbol("vx", i)
        vs = Symbol("vs", i)
        simd_body = quote
            $simd_body
            $vx = vgather(Xs[$i], idx, nomask, $align)
            $vs = vload(Vec{$N, $Tv}, nzvs[$i], j, $align)
            $vacc = muladd($vs, $vx, $vacc)
        end
        reduce_vaccs = quote
            $reduce_vaccs
            $acc = sum($vacc)
        end
        scalar_body = quote
            $scalar_body
            $acc = muladd(nzvs[$i][j], Xs[$i][nzind[j]], $acc)
        end
        push!(mul_diag_exprs, quote
            diags[$i][col] * $acc
        end)
    end
    mul_diag = Expr(:call, :+, mul_diag_exprs...)

    quote
        Y, β = Yβ
        if β != 1
            β != 0 ? rmul!(Y, β) : fill!(Y, zero(eltype(Y)))
        end

        S1 = parent(triplets[1][2])
        nzind = rowvals(S1)
        Ti = eltype(nzind)
        nomask = Vec($(ntuple(_ -> true, N)))

        nzvs = ($((:(nonzeros(parent(triplets[$i][2]))) for i in 1:nt)...),)
        diags = ($((:(triplets[$i][1].diag) for i in 1:nt)...),)

        for k = 1:size(Y, 2)
            Xs = ($((:(unsafe_column(triplets[$i][3], k)) for i in 1:nt)...),)
            @inbounds for col = 1:S1.n
                nzr = nzrange(S1, col)
                simd_end = last(nzr) - $N + 1
                j = first(nzr)
                $init_vaccs
                @inbounds while j <= simd_end
                    $simd_body
                    j += $N
                end
                $reduce_vaccs

                @inbounds while j <= last(nzr)
                    $scalar_body
                    j += 1
                end

                Y[col, k] += $mul_diag
            end
        end
        return Y
    end
end
