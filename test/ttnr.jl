@testset "TNO" begin
    local_tensor = TensorMap(
        reshape(collect(1.0:64.0), 2, 2, 2, 2, 2, 2),
        ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'
    )

    tno = TNO(local_tensor; unitcell = (2, 3))
    @test size(tno) == (2, 3)
    @test tno isa AbstractMatrix
    @test all(tno[i, j] ≈ local_tensor for i in axes(tno, 1), j in axes(tno, 2))

    other = copy(local_tensor)
    tno[1, 2] = other
    @test tno[1, 2] === other

    unitcell = [copy(local_tensor) for _ in 1:2, _ in 1:2]
    tno2 = TNO(unitcell)
    @test size(tno2) == (2, 2)
    @test tno2[2, 1] ≈ local_tensor

    @test_throws ArgumentError TNO(local_tensor; unitcell = (0, 2))
    @test_throws ArgumentError TNO(reshape(typeof(local_tensor)[], 0, 1))

    tno_copy = copy(tno2)
    @test tno_copy isa TNO
    @test tno_copy !== tno2
    @test tno_copy[1, 1] === tno2[1, 1]
    @test tno_copy.A !== tno2.A
end

@testset "TNO apply!" begin
    local_tensor = TensorMap(
        reshape(collect(1.0:64.0), 2, 2, 2, 2, 2, 2),
        ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'
    )

    top = TNO(local_tensor; unitcell = (2, 2))
    bottom = TNO(local_tensor; unitcell = (2, 2))
    merged = apply!(top, bottom, truncrank(8))

    @test size(merged) == (2, 2)
    @test merged isa TNO
    @test space(merged[1, 1], 1) == space(top[1, 1], 1)
    @test space(merged[1, 1], 2) == space(bottom[1, 1], 2)

    bad_tensor = TensorMap(
        reshape(collect(1.0:128.0), 2, 4, 2, 2, 2, 2),
        ℂ^2 ⊗ (ℂ^4)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'
    )
    @test_throws ArgumentError apply!(TNO(bad_tensor; unitcell = (2, 2)), bottom, truncrank(8))

    scheme_top = ThermalTNR(TNO(local_tensor; unitcell = (1, 1)))
    scheme_bottom = ThermalTNR(TNO(local_tensor; unitcell = (1, 1)))
    @test apply!(scheme_top, scheme_bottom, truncrank(8)) isa ThermalTNR
end

@testset "ThermalTNR finalize!" begin
    local_tensor = TensorMap(
        reshape(collect(1.0:64.0), 2, 2, 2, 2, 2, 2),
        ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'
    )

    tno = TNO([copy(local_tensor) for _ in 1:2, _ in 1:2])
    scheme = ThermalTNR(tno)
    n = finalize!(scheme)

    @test isfinite(n)
    @test n > 0
    @test n ≈ norm(@tensor local_tensor[1 1; 2 3 2 3])
    @test all(norm(@tensor scheme.T[i, j][1 1; 2 3 2 3]) ≈ 1 for i in 1:2, j in 1:2)
end

@testset "ThermalTNR run!" begin
    local_tensor = TensorMap(
        reshape(collect(1.0:64.0), 2, 2, 2, 2, 2, 2),
        ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'
    )

    scheme = ThermalTNR(TNO(local_tensor; unitcell = (1, 1)))
    layer = TNO(local_tensor; unitcell = (1, 1))
    data = run!(scheme, layer, truncrank(8), maxiter(1))

    @test data isa Vector{Float64}
    @test length(data) == 2
    @test all(isfinite, data)
    @test all(n -> n > 0, data)
    @test norm(@tensor scheme.T[1, 1][1 1; 2 3 2 3]) ≈ 1
end
