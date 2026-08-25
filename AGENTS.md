# AGENTS.md

Guidance for coding agents working in the TNRKit.jl repository.

## What this package is

TNRKit.jl implements tensor network renormalization (TNR) schemes for 2D and 3D
classical lattice models, built on [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)
(symmetric tensors) and MatrixAlgebraKit.jl (decompositions / truncation strategies).
It computes partition functions, free energies, CFT data and central charges.

Julia ≥ 1.11. Package name `TNRKit`, currently v0.7.0, upstream is
`QuantumKitHub/TNRKit.jl` (`master` is the default branch).

## Repository layout

```
src/TNRKit.jl          module file: every include + every export lives here
src/schemes/           coarse-graining algorithms
  tnrscheme.jl         abstract TNRScheme{E,S}, Finalizer, run! driver
  trg.jl btrg.jl hotrg.jl atrg.jl looptnr.jl symmetric_looptnr.jl
  hotrg3d.jl atrg3d.jl ttnr.jl
  impuritytrg.jl impurityhotrg.jl correlationhotrg.jl
  ctm/                 corner transfer matrix methods (square, triangular, honeycomb)
src/models/            tensor constructors: ising, potts, clock, XY, sixvertex,
                       phi4_real, phi4_complex, gross-neveu, quantum_1D, ising_{triangular,honeycomb}
src/utility/           free_energy, cft, transfer_matrix, stopping, finalize,
                       projectors, entropies, blocking, cdl, network_value,
                       gs_degeneracy, structuredvector
test/{schemes,models,misc,fermions}/
docs/                  Documenter + DocumenterCitations
examples/example.jl
```

`src/TNRKit.jl` is the single source of truth for the public API — anything new must be
`include`d **and** `export`ed there, or it is invisible to users and to the docs.

## Core abstractions

A scheme is a mutable struct subtyping `TNRScheme{E,S}` (`E` = scalar type, `S` = space type)
that holds the tensor(s) being coarse-grained. Three methods make it work:

- `step!(scheme, trunc::TruncationStrategy)` — one coarse-graining iteration, mutates in place, returns the scheme.
- `finalize!(scheme)` — normalizes the tensor(s) and returns the normalization factor (in [src/utility/finalize.jl](src/utility/finalize.jl)).
- `Base.show(io, scheme)` — one-line header plus a `summary` of each field.

`run!` ([src/schemes/tnrscheme.jl](src/schemes/tnrscheme.jl)) is the shared driver:

```julia
run!(scheme, trscheme::TruncationStrategy, criterion::stopcrit,
     finalizer::Finalizer = default_Finalizer;
     finalize_beginning = true, verbosity = 1)
```

It returns a `Vector{E}` of finalizer outputs. A `Finalizer(f!, E)` pairs the per-step
function with its output type so the result vector is concretely typed — if a new
finalizer returns something other than `Float64`, construct it with the right `E`
(see `ImpurityTRG_Finalizer`, `ImpurityHOTRG_Finalizer`).

Stopping criteria (`maxiter`, `convcrit`, composable with `&`) live in
[src/utility/stopping.jl](src/utility/stopping.jl); each new `stopcrit` needs a
`stopping_info` method so the end-of-run log explains why it stopped.

Verbosity is `LoggingExtras.withlevel` + `@infov`: 0 = silent, 1 = start/end, 2 = per step.
Do not add bare `println`/`@info` to library code — use `@infov n`.

## Leg conventions (get these wrong and everything is silently wrong)

2D schemes take a tensor in `V₁ ⊗ V₂ ← V₃ ⊗ V₄` with legs ordered

```
     3
     |
     v
     |
1-<--┼--<-4
     |
     v
     |
     2
```

3D schemes take `V_D ⊗ V_U′ ← V_N ⊗ V_E ⊗ V_S′ ⊗ V_W′` (Down, Up, North, East, South, West).
CTM schemes have their own conventions — `c4vCTM`, for example, accepts either a `(2,2)`
"flipped-arrow" tensor (W, S, N, E) or a `(0,4)` "unflipped" tensor (N, E, S, W); read the
docstring of the specific scheme before touching index order.

Any new model constructor must produce tensors in these conventions.

## Conventions to follow

- Contractions use `@tensor` / `@plansor` from TensorKit; prefer `transpose`/`permute` with
  explicit index tuples over manual reshaping.
- Schemes are parametrically typed on the concrete tensor type
  (`TT <: AbstractTensorMap{E,S,2,2}`) and constructed through an inner constructor —
  keep the code type-stable, and never hardcode `Float64` where `scalartype(T)` is meant
  (see commit 42d80a9).
- Truncation is a MatrixAlgebraKit `TruncationStrategy` (`truncrank(16)`, `trunctol(...)`,
  composable with `&`) — never invent a bespoke truncation argument.
- Docstrings use DocStringExtensions (`$(TYPEDEF)`, `$(TYPEDFIELDS)`, `$(FUNCTIONNAME)`,
  `$(SIGNATURES)`) and follow the existing template: description, `# Constructors`,
  `# Running the algorithm`, a `!!! info "verbosity levels"` admonition, `# Fields`,
  `# References` with a `[...](@cite key)`. Citation keys go in
  [docs/src/assets/tnrkit.bib](docs/src/assets/tnrkit.bib).
- Docs are `@autodocs` over the whole module ([docs/src/lib/lib.md](docs/src/lib/lib.md)),
  so an exported symbol without a docstring degrades the manual immediately. New schemes
  should also be listed in [docs/src/index.md](docs/src/index.md) and [README.md](README.md).
- Unicode identifiers are used freely (`β`, `τ₀`, `ϕ`, `ising_βc`); match the surrounding
  naming rather than ASCII-ifying.

## Building and testing

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'                        # full suite
julia --project -e 'using Pkg; Pkg.test(test_args=["schemes"])'    # one test group
julia --project -e 'using Pkg; Pkg.test(test_args=["--fast"])'     # reduced set (CI uses this on draft PRs)
julia --project=docs docs/make.jl                                  # build docs
```

Tests run through [ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl):
every file under `test/` is auto-discovered and run in its own isolated process, so
`test/runtests.jl` contains **no** `include` statements — adding `test/foo/foo.jl` is enough,
and each such file must be self-contained (`using Test, TNRKit, TensorKit, ...` at the top).
Remaining `test_args` filter which files run; `--jobs=N`, `--list`, `--quickfail` are also accepted.
A `fast_tests` constant is injected into every worker for gating slow cases.

Scheme tests are physics regression tests: run a scheme on the (usually critical) Ising
model and compare `free_energy(data, β)` against the exact Onsager value with an `rtol`.
New schemes belong in `test/schemes/schemes.jl` in that same form; new models belong in the
`model_temp_answer_string_*` tables in `test/models/models.jl` with a reference free energy.
Tolerances are tuned per scheme — if a test fails, investigate before loosening `rtol`.

## CI and formatting

- `Tests` (CI.yml), `Format` (FormatCheck.yml) and `Documentation` all delegate to the shared
  `QuantumKitHub/QuantumKitHubActions` workflows. There is no formatter config in this repo,
  so the shared job decides — do not add a local `.JuliaFormatter.toml` on a whim.
- Observed style: 4-space indent, spaces around `=` in keyword arguments
  (`truncrank(24), rtol = 2.0e-6`, `; h = 0`), lowercase scientific literals (`1.0e-14`),
  trailing commas in multi-line argument lists, multi-line signatures indented one level.
  Match the file you are editing.
- Work happens on branches and lands via PR to `master`; commit subjects are short and
  imperative ("Implement Thermal TNR", "Fix docbuild"). Do not commit or push unless asked.
- `Manifest.toml` is checked in; leave it alone unless the task is a dependency bump, in
  which case update `[compat]` in [Project.toml](Project.toml) too.

## Gotchas

- The package is under active development and the interface is explicitly declared unstable —
  breaking changes are acceptable, but they need a version bump and README/docs updates.
- `free_energy(data, β; scalefactor = 2.0, initial_size = 1.0)` assumes the network volume
  rescales by `scalefactor` per step. The default `2.0` is right for TRG/BTRG/LoopTNR;
  HOTRG and ATRG need `scalefactor = 4.0`, the 3D schemes need `8.0`. Models whose initial
  tensor holds more than one site (e.g. `classical_ising_impurity`, the triangular/honeycomb
  Ising tensors) additionally need `initial_size`. Getting either wrong yields a free energy
  that is off in a way that looks like a convergence problem, so check the tests in
  [test/schemes/schemes.jl](test/schemes/schemes.jl) for the value a given scheme uses.
- `run!` normalizes once before the first step by default (`finalize_beginning = true`), so
  `length(data) == steps + 1`. Off-by-one errors in free-energy comparisons usually trace here.
- Zygote/OptimKit are used only by the loop-optimization schemes (`LoopTNR`, `SLoopTNR`);
  changes there can silently break AD — run the LoopTNR tests specifically.
