module TestAmbiguity
using Test

@testset "method ambiguity" begin
    # Ambiguity test is run inside a clean process.
    # https://github.com/JuliaLang/julia/issues/28804
    script = joinpath(@__DIR__, "detect_ambiguities.jl")
    code = """
    $(Base.load_path_setup_code())
    include($(repr(script)))
    """
    cmd = Base.julia_cmd()
    if Base.JLOptions().color == 1
        cmd = `$cmd --color=yes`
    end
    cmd = `$cmd --startup-file=no -e $code`
    @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
end

end  # module
