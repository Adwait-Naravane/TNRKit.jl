"""
    TNOTensor{E,S}

Type alias for a local tensor-network-operator tensor with two physical legs and four
virtual legs, i.e. an `AbstractTensorMap{E,S,2,4}`.

Following the PEPO convention used in PEPSKit, the local tensor is interpreted as
`P_out ⊗ P_in <- N ⊗ E ⊗ S ⊗ W`.
"""
const TNOTensor{E, S} = AbstractTensorMap{E, S, 2, 4}

"""
    $(TYPEDEF)

Matrix-like container for a tensor network operator (TNO) unit cell.

The entries are local operator tensors with two physical indices and four virtual
indices. The container behaves like an `AbstractMatrix`, so it supports `size`,
`axes`, indexing and iteration.

### Constructors
    $(FUNCTIONNAME)(A::AbstractMatrix{<:TNOTensor})
    $(FUNCTIONNAME)(A::TNOTensor; unitcell=(1, 1))

### Fields

$(TYPEDFIELDS)
"""
struct TNO{E, S, TT <: TNOTensor{E, S}} <: AbstractMatrix{TT}
    "Matrix of local TNO tensors over the unit cell."
    A::Matrix{TT}

    function TNO(A::AbstractMatrix{TT}) where {E, S, TT <: TNOTensor{E, S}}
        return new{E, S, TT}(Matrix(A))
    end
end

function TNO(A::TT; unitcell::Tuple{Int, Int} = (1, 1)) where {E, S, TT <: TNOTensor{E, S}}
    rows, cols = unitcell
    rows > 0 || throw(ArgumentError("The unit cell must have a positive number of rows."))
    cols > 0 || throw(ArgumentError("The unit cell must have a positive number of columns."))
    return TNO([copy(A) for _ in 1:rows, _ in 1:cols])
end

Base.IndexStyle(::Type{<:TNO}) = IndexCartesian()
Base.size(tno::TNO) = size(tno.A)
Base.axes(tno::TNO) = axes(tno.A)
Base.getindex(tno::TNO, i::Int, j::Int) = tno.A[i, j]
Base.setindex!(tno::TNO, value, i::Int, j::Int) = setindex!(tno.A, value, i, j)

"""
    $(TYPEDEF)

Minimal storage object for thermal tensor network renormalization on a square-lattice
TNO.

### Constructors
    $(FUNCTIONNAME)(T::TNO)

### Fields

$(TYPEDFIELDS)
"""
mutable struct ThermalTNR{E, S} <: TNRScheme{E, S}
    "Tensor network operator stored in the current TTNR layer."
    T::TNO{E, S}

    function ThermalTNR(T::TT) where {E, S, TT <: TNO{E, S}}
        return new{E, S}(T)
    end
end

const _TNO_NORTH_AXIS = 3
const _TNO_EAST_AXIS = 4
const _TNO_SOUTH_AXIS = 5
const _TNO_WEST_AXIS = 6

@inline _right_index(j, ncols) = mod1(j + 1, ncols)
@inline _down_index(i, nrows) = mod1(i + 1, nrows)
@inline _up_index(i, nrows) = mod1(i - 1, nrows)

function _check_tno_bond_structure(tno::TNO)
    nrows, ncols = size(tno)

    for i in 1:nrows, j in 1:ncols
        T = tno[i, j]
        T_right = tno[i, _right_index(j, ncols)]
        T_down = tno[_down_index(i, nrows), j]

        space(T, _TNO_EAST_AXIS) == space(T_right, _TNO_WEST_AXIS)' ||
            throw(ArgumentError("East-west bond mismatch between sites ($i, $j) and ($i, $(_right_index(j, ncols)))."))

        space(T, _TNO_SOUTH_AXIS) == space(T_down, _TNO_NORTH_AXIS)' ||
            throw(ArgumentError("North-south bond mismatch between sites ($i, $j) and ($(_down_index(i, nrows)), $j)."))
    end

    return nothing
end

function _check_tno_compatibility(top::TNO, bottom::TNO)
    size(top) == size(bottom) ||
        throw(ArgumentError("The two TNOs must have the same unit-cell dimensions."))

    _check_tno_bond_structure(top)
    _check_tno_bond_structure(bottom)

    for i in axes(top, 1), j in axes(top, 2)
        T_top = top[i, j]
        T_bottom = bottom[i, j]

        space(T_top, 2) == space(T_bottom, 1)' ||
            throw(ArgumentError("Physical output/input mismatch at site ($i, $j)."))
    end

    return nothing
end

function QR_two_pepo_left(O1::TNOTensor, O2::TNOTensor, ind::Int)
    pb = (1, ind)
    p, q1 = ind_pair(O1, pb)
    _, Rb = left_orth(permute(O1, (q1, p)))

    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    _, Rt = left_orth(permute(O2, (q2, p)))

    @tensor M[-1 -2; -3 -4] := Rt[-3; 1 -1] * Rb[-4; 1 -2]
    _, R = left_orth(permute(M, ((3, 4), (1, 2))))
    return R
end

function QR_two_pepo_right(O1::TNOTensor, O2::TNOTensor, ind::Int)
    pb = (1, ind)
    p, q1 = ind_pair(O1, pb)
    Rb, _ = right_orth(permute(O1, (p, q1)))

    pt = (2, ind)
    p, q2 = ind_pair(O2, pt)
    Rt, _ = right_orth(permute(O2, (p, q2)))

    @tensor M[-1 -2; -3 -4] := Rt[1 -1; -3] * Rb[1 -2; -4]
    R, _ = right_orth(permute(M, ((1, 2), (3, 4))))
    return R
end

function QR_two_pepo(O1::TNOTensor, O2::TNOTensor, ind::Int; side = :left)
    if side == :left
        return QR_two_pepo_left(O1, O2, ind)
    elseif side == :right
        return QR_two_pepo_right(O1, O2, ind)
    else
        throw(ArgumentError("side should be :left or :right"))
    end
end

function QR_two_pepo(
        O1::TNOTensor, O2::TNOTensor, O3::TNOTensor, O4::TNOTensor,
        ind1::Int, ind2::Int; side = :left
    )
    if side == :left
        return QR_two_pepo_left(O1, O2, ind1)
    elseif side == :right
        return QR_two_pepo_right(O3, O4, ind2)
    else
        throw(ArgumentError("side should be :left or :right"))
    end
end

function R1R2(
        A1::TNOTensor, A2::TNOTensor, A3::TNOTensor, A4::TNOTensor,
        ind1::Int, ind2::Int; check_space = true
    )
    RA1 = QR_two_pepo(A1, A2, A3, A4, ind1, ind1)
    RA2 = QR_two_pepo(A1, A2, A3, A4, ind2, ind2; side = :right)
    if check_space && domain(RA1) != codomain(RA2)
        throw(ArgumentError("space mismatch"))
    end
    return RA1, RA2
end

function find_P1P2(
        A1::TNOTensor, A2::TNOTensor, A3::TNOTensor, A4::TNOTensor,
        p1::Int, p2::Int, trunc::TruncationStrategy; check_space = true
    )
    R1, R2 = R1R2(A1, A2, A3, A4, p1, p2; check_space = check_space)
    return oblique_projector(R1, R2, trunc)
end

function _bond_projectors(top::TNO, bottom::TNO, trunc::TruncationStrategy)
    nrows, ncols = size(top)
    north_proj = Matrix{Any}(undef, nrows, ncols)
    east_proj = Matrix{Any}(undef, nrows, ncols)
    south_proj = Matrix{Any}(undef, nrows, ncols)
    west_proj = Matrix{Any}(undef, nrows, ncols)

    for i in 1:nrows, j in 1:ncols
        inorth = _up_index(i, nrows)
        jeast = _right_index(j, ncols)

        Pnorth, Psouth = find_P1P2(
            top[i, j], bottom[i, j], top[inorth, j], bottom[inorth, j], 3, 5, trunc
        )
        Peast, Pwest = find_P1P2(
            top[i, j], bottom[i, j], top[i, jeast], bottom[i, jeast], 4, 6, trunc
        )

        north_proj[i, j] = Pnorth
        south_proj[inorth, j] = Psouth
        east_proj[i, j] = Peast
        west_proj[i, jeast] = Pwest
    end

    return north_proj, east_proj, south_proj, west_proj
end

function _compose_local_tno(top::TNOTensor, bottom::TNOTensor, Pnorth, Peast, Psouth, Pwest)

    @tensor merged[-1 -2; -3 -4 -5 -6] :=
        top[1 -2; 7 8 9 10] *
        bottom[-1 1; 3 4 5 6] *
        Pwest[-6; 6 10] *
        Pnorth[3 7; -3] *
        Peast[4 8; -4] *
        Psouth[-5; 5 9]

    return merged
end

"""
    apply!(top::TNO, bottom::TNO, trunc::TruncationStrategy)

Compose two TNO layers sitewise. The physical input leg of `top` is contracted with the
physical output leg of `bottom`, after which each pair of corresponding virtual bonds is
merged into a single fused bond.

The result is returned as a new `TNO` with the same unit-cell shape.
"""
function apply!(top::TNO, bottom::TNO, trunc::TruncationStrategy)
    _check_tno_compatibility(top, bottom)
    north_proj, east_proj, south_proj, west_proj = _bond_projectors(top, bottom, trunc)
    merged = [
        _compose_local_tno(
                top[i, j], bottom[i, j],
                north_proj[i, j], east_proj[i, j], south_proj[i, j], west_proj[i, j]
            ) for
            i in axes(top, 1), j in axes(top, 2)
    ]
    return TNO(merged)
end

function apply!(top::ThermalTNR, bottom::ThermalTNR, trunc::TruncationStrategy)
    top.T = apply!(top.T, bottom.T, trunc)
    return top
end
