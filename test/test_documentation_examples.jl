using Test, FeastKit, LinearAlgebra, SparseArrays, Markdown

# Execute the source shown to readers, rather than a separately maintained copy.
# Locate a Julia fence after a stable section heading; respect longer outer
# fences so docstrings can contain their own triple-backtick examples.
function documentation_block(file, heading)
    text = read(joinpath(@__DIR__, "..", "docs", "src", file), String)
    section = findfirst(heading, text)
    section === nothing && error("Missing documentation section: $heading")
    lines = split(SubString(text, last(section)+1), '\n')
    opening = findfirst(line -> occursin(r"^\s*`{3,}julia\s*$", line), lines)
    opening === nothing && error("Missing Julia example: $heading")
    fence = match(r"^\s*(`{3,})", lines[opening]).captures[1]
    closing = findnext(line -> strip(line) == fence, lines, opening+1)
    closing === nothing && error("Unclosed Julia fence: $heading")
    return join(lines[opening+1:closing-1], '\n')
end

function documentation_sandbox()
    sandbox = Module(gensym(:DocumentationExample))
    Core.eval(sandbox, :(using FeastKit, LinearAlgebra, SparseArrays))
    Core.eval(sandbox, quote
        A = Matrix(Diagonal(Float64.(1:40)))
        B = Matrix{Float64}(I,40,40)
        Emin, Emax = 0.5, 3.5
        interval = (Emin,Emax)
        M0 = 10
    end)
    return sandbox
end

@testset "Executable documentation regressions" begin
    @testset "Convergence workflow" begin
        sandbox = documentation_sandbox()
        code = documentation_block("getting_started.md", "**Problem**: FeastKit isn't converging well")
        Base.include_string(sandbox, code)
        for name in (:result, :result_zolotarev)
            r = getfield(sandbox,name)
            @test r.info == 0
            @test r.lambda ≈ [1.,2.,3.] atol=1e-8
        end
        @test !occursin("2=detailed",code)
    end
    @testset "Integration benchmark" begin
        sandbox = documentation_sandbox()
        code = documentation_block("performance.md", "### Integration Method Selection")
        # Keep the documented function intact; run it on a small test fixture.
        Base.include_string(sandbox, first(split(code,"# Run benchmark")))
        rows = Core.eval(sandbox, :(benchmark_integration_methods(A,interval)))
        @test length(rows) == 12
        @test all(row -> row[4] == 3 && row[5] == 0, rows)
    end
    @testset "Distribution settings" begin
        sandbox = documentation_sandbox()
        Base.include_string(sandbox, documentation_block("performance.md", "### Eigenvalue Distribution Optimization"))
        for kind in ("clustered","sparse","uniform")
            r = Core.eval(sandbox, :(optimize_for_distribution(A,interval,$kind)))
            @test r.info == 0
            @test r.lambda ≈ [1.,2.,3.] atol=1e-8
        end
    end
    @testset "Spurious eigenvalues" begin
        sandbox = documentation_sandbox()
        Base.include_string(sandbox, documentation_block("custom_contours.md", "!!! warning \"Issue: Spurious eigenvalues\""))
        r = getfield(sandbox,:result)
        @test r.info == 0
        @test r.lambda ≈ [1.,2.,3.] atol=1e-8
    end
    @testset "Parallel API" begin
        sandbox = documentation_sandbox()
        Base.include_string(sandbox, documentation_block("api_reference.md", "### ParallelFeastState"))
        @test getfield(sandbox,:state) isa ParallelFeastState{Float64}
        @test getfield(sandbox,:result).info == 0
    end
    @testset "Detailed parallel benchmark" begin
        sandbox = documentation_sandbox()
        # Execute the detailed benchmark call on the same small fixture.
        code = documentation_block("parallel_computing.md", "### Benchmarking")
        detailed = match(r"(?m)^.*pfeast_rci_benchmark\(.*$",code)
        @test detailed !== nothing
        Base.include_string(sandbox,detailed.match)
    end
    @testset "Contributor snippets and nested fences" begin
        for (file,heading) in (("contributing.md","### Docstrings"),
                               ("developer_guide.md","### Documentation"),
                               ("developer_guide.md","### Type Stability"))
            code = documentation_block(file,heading)
            sandbox = Module(gensym(:ContributorExample))
            Core.eval(sandbox, :(using LinearAlgebra))
            Base.include_string(sandbox,code)
            name = heading == "### Type Stability" ? :solve :
                   file == "contributing.md" ? :my_function : :feast_sygv!
            @test getfield(sandbox,name) isa Function
            if heading == "### Type Stability"
                @test Core.eval(sandbox, :(solve(Matrix{Float64}(I,2,2),ones(2)))) == ones(2)
            else
                # Check the Markdown parser as well as Julia: the embedded
                # triple fence must stay inside one outer code block.
                page = Markdown.parse(read(joinpath(@__DIR__,"..","docs","src",file),String))
                blocks = filter(x -> x isa Markdown.Code, page.content)
                @test any(b -> occursin("function $name",b.code) && occursin("```julia",b.code),blocks)
            end
        end
    end
end
