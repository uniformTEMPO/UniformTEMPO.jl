# compute a process tensor for spin boson, convert to choi matrix and check for quantum memory

using UniformTEMPO

pt = uniTEMPO([0 1; 1 0], 0.1, t -> exp(-0.5 * abs(t) - im * t), 1e-8);

h_sys = [1 0; 0 -1]

t_1 = 10;
t_2 = 20;
proc = process_tensor(pt, [t_1, t_2]; h_s=h_sys)# generate single intervention process tensor

χ = pt_to_choi(proc) # convert to choi matrix

# check that its a valid choi state
using LinearAlgebra
pχ = eigvals(reshape(χ, 16, 16))
sum(pχ)

# ppt criterion to check for nonseparability
function partial_transpose(x, dims)
    x = reshape(x, dims[1], dims[2], dims[1], dims[2])
    xpt = permutedims(x, [3, 2, 1, 4])
    return reshape(xpt, dims[1] * dims[2], dims[1] * dims[2])
end

χ_pt = partial_transpose(χ, (4, 4))
minimum(real.(eigvals(χ_pt)))
# process tensor is entangled -> there is definitely a quantum bit in the environment!

# get the stationary process tensor
x0 = steadystate(pt; h_s=h_sys, return_full=true) # get steady state of system + environment
proc = process_tensor(pt, x0, [0, t_2 - t_1]; h_s=h_sys) # generate stationary process tensor

χ = pt_to_choi(proc) # convert to choi matrix
χ_pt = partial_transpose(χ, (2, 4))
minimum(real.(eigvals(χ_pt)))
# stationary single-intervention process tensor is PPT