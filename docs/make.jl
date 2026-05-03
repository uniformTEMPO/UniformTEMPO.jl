using Documenter, UniformTEMPO

makedocs(
sitename="UniformTEMPO.jl",
authors="Valentin Link, Konrad Mickiewicz, Matteo Garbellini",
pages=["Home"=>"index.md",
"Introduction"=>"introduction.md",
"Advanced"=>"advanced.md",
"Examples" => [
    "Example problems" => [
        "Computing correlation functions" => "examples/correlation_functions.md",
        "Floquet dynamics" => "examples/floquet_dynamics.md",
        "Computing process tensors" => "examples/process_tensors.md"
    ]
],
"Convergence guidelines"=>"convergence.md",
"Troubleshooting"=>"troubleshooting.md",
"API Reference"=>"reference.md",
"Authors and Citation"=>"authors.md"]
)

deploydocs(
    repo = "github.com/uniformTEMPO/UniformTEMPO.jl.git",
    devbranch = "main"
)

# local hosting for development
# using LiveServer
# serve(dir=joinpath(@__DIR__, "build"), port=8000)