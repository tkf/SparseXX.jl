### mul!

for TA in [SparseXXMatrixCSC, AdjOrTrans{<:Any, <:SparseXXMatrixCSC}]
    @eval LinearAlgebra.mul!(C::StridedVecOrMat, A::$TA, B::StridedVecOrMat) =
        mul_api!(C, A, B, one(eltype(B)), zero(eltype(C)))
    @eval LinearAlgebra.mul!(C::StridedVecOrMat, A::$TA, B::StridedVecOrMat,
                             α::Number, β::Number) =
        mul_api!(C, A, B, α, β)
end

function mul_api!(C, A, B, α, β)
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

function mul_simd!(C, A::SparseXXMatrixCSC, B, α, β,
                   ::Val{N} = Val(4),
                   ::Val{align} = Val(false),
                   ) where {N, align}
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end

    nomask = Vec(ntuple(_ -> true, N))
    for k = 1:size(C, 2)
        Ck = SubArray(C, (Base.Slice(Base.OneTo(size(C, 1))), k))
        @inbounds for col = 1:A.n
            αxj = B[col, k] * α
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

function mul_simd!(C, adjA::AdjOrTrans, B, α, β)
    # eltype(A) is a SIMD.ScalarTypes hence is a Real; no need for
    # adjoint for each element
    A = adjA.parent
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            a = unsafe_column(A, col)
            b = SubArray(B, (Base.Slice(Base.OneTo(size(B, 1))), k))
            C[col, k] += α * dot_simd(a, b)
        end
    end
    return C
end


### fmul!

"""
    fmul!((Y, β), L, M, R)

Fused multiplication (and addition).

```math
Y = L M R + Y β
```

Following combinations are planned:

| ``L``        | ``M``        | ``R``        | done? |
|:---          |:---          |:---          |:---   |
| diagonal     | spmat'       | vec/mat      | [ ]   |
| spmat        | diagonal     | vec/mat      | [ ]   |
| spmat        | diagonal     | spvec/spmat' | [ ]   |
| spmat'       | diagonal     | smat         | [ ]   |

where

* diagonal: `Diagonal`, `UniformScaling`, or a `Number`
* spmat: `SparseXXMatrixCSC`
* spmat': `Adjoint` or `Transpose` of `SparseXXMatrixCSC`
* vec/mat: `AbstractVecOrMat`

"""
function fmul!(Yβ, L, M, R)
    notimplemented(fmul!, Yβ, L, M, R)
end
