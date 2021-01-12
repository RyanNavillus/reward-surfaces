import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gym import spaces
from scipy.sparse.linalg import LinearOperator, eigsh

def gradtensor_to_npvec(params, include_bn=True):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in params if filter(p)])


def npvec_to_tensorlist(vec, params, device):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).to(device).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def calculate_est_hesh_eigenvalues(model, num_samples, tol):
    model.setup_buffer(num_samples)

    model.dot_prod_calcs = 0

    def hess_vec_prod(vec):
        model.dot_prod_calcs += 1
        vec = npvec_to_tensorlist(vec, model.parameters(), model.device)
        model.calculate_hesh_vec_prod(vec, num_samples)
        return gradtensor_to_npvec(model.parameters())


    N = sum(np.prod(param.shape) for param in model.parameters())
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=tol)
    maxeig = eigvals[0]
    print(f"max eignvalue = {maxeig}")
    print(eigvecs[0])
    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*.51
    def shifted_hess_vec_prod(vec):
        return hess_vec_prod(vec) - shift*vec

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, tol=tol)
    eigvals = eigvals + shift
    mineig = eigvals[0]
    print(f"min eignvalue = {mineig}")

    assert maxeig >= 0 or mineig < 0, "something weird is going on but this case is handled by the loss landscapes paper, so duplicating that here"

    print("number of evaluations required: ", model.dot_prod_calcs)

    model.cleanup_buffer()

    return float(maxeig), float(mineig)
