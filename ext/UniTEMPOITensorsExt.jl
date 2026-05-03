module UniTEMPOITensorsExt

using UniformTEMPO
using ITensors
using LinearAlgebra


"""
    UniformPTMPOITensor

A struct representing a UniformPTMPO but with ITensors. System indices are in Liouville space by default via Julias generic column-major ordering.
To convert the Liouville index to a pair of site indices use the `cmb` tensor, i.e. `q_hilbert_space = pt.q * pt.cmb * pt.cmb`.
"""
struct UniformPTMPOITensor
    s_dim::Int
    delta_t::Real
    q::ITensor
    v_l::ITensor
    v_r::ITensor
    cmb::ITensor
end

function UniformTEMPO.to_ITensor(pt::UniformPTMPO; tags=[])
    i_s = Index(pt.s_dim^2, tags=join(["LiouvilleSite", tags...], ","))
    i_s
    i_a = Index(bond_dim(pt), tags=join(["AuxBathSite", tags...], ","))
    q_it = ITensor(pt.q, i_a', i_s', i_a, i_s)
    v_l_it = ITensor(pt.v_l[:], i_a)
    v_r_it = ITensor(pt.v_r[:], i_a)
    i_sp = Index(pt.s_dim, tags=join(["Site+", tags...], ","))
    i_sm = Index(pt.s_dim, tags=join(["Site-", tags...], ","))
    cmb = ITensor(reshape(Matrix(I, pt.s_dim^2, pt.s_dim^2), pt.s_dim^2, pt.s_dim, pt.s_dim), i_s, i_sp, i_sm)
    return UniformTEMPO.UniformPTMPOITensor(pt.s_dim, pt.delta_t, q_it, v_l_it, v_r_it, cmb)
end

function __init__()
    @eval UniformTEMPO begin
        const UniformPTMPOITensor = $(UniTEMPOITensorsExt.UniformPTMPOITensor)
        export UniformPTMPOITensor
    end
end

end