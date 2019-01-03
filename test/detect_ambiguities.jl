using Test

_deps = quote
    SparseArrays
    LinearAlgebra
    FillArrays
    SIMD
end
deps = [x for x in _deps.args if x isa Symbol]
@debug "Loading dependencies:" deps
for m in deps
    @eval using $m
end

modules = eval.(deps)
ambiguities = detect_ambiguities(modules..., imported=true, recursive=true)
if !isempty(ambiguities)
    @error "Ambiguities found in dependencies." deps ambiguities
else
    using SparseXX
    @test detect_ambiguities(SparseXX, imported=true, recursive=true) == []
end
