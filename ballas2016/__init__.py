import torch
import torch.nn as nn
import torch.nn.functional as F
import ballas2016.gru_rcn_cpp
from torch.nn.init import xavier_uniform_ as xavier_uniform
from typing import Tuple


# region CPU implementation
class GRU_RCN_CPP_Function(torch.autograd.Function):
    """
    Wrapper function around the forward and backward C++ implementations for running on the CPU.
    """
    @staticmethod
    def forward(ctx, x, W, U, C, h):
        new_h, *cache = gru_rcn_cpp.forward(x, W, U, C, h)
        ctx.save_for_backward(W, U, C, *cache, x, h)

        return new_h

    @staticmethod
    def backward(ctx, grad_h):
        weights_and_cache = ctx.saved_tensors
        dx, dh, d_wz, d_wr, d_wc, d_uz, d_ur, d_uc = gru_rcn_cpp.backward(grad_h, *weights_and_cache)

        W_grad = torch.cat([d_wz, d_wr, d_wc], dim=0)
        U_grad = torch.cat([d_uz, d_ur], dim=0)
        return dx, W_grad, U_grad, d_uc, dh
# endregion


# region GRU-RCN layer wrapper
class GRU_RCN(torch.nn.Module):
    """
    The convolutional GRU layer from the paper "Delving deeper into convolutional networks for learning video
    representations". This class wraps the C++ and CUDA implementations depending on whether a GPU is available or not.
    """
    def __init__(self, in_channels: int, nb_kernels: int, kernel_size: Tuple[int, int], batch_first: bool = True):
        super(GRU_RCN, self).__init__()
        self.Ox = in_channels
        self.Oh = nb_kernels

        if not batch_first:
            raise NotImplementedError("Batch dimension needs to be first at the moment!")

        self.W = nn.Parameter(torch.empty(3 * self.Oh, self.Ox, *kernel_size), requires_grad=True)
        self.U = nn.Parameter(torch.empty(2 * self.Oh, self.Oh, *kernel_size), requires_grad=True)
        self.C = nn.Parameter(torch.empty(1 * self.Oh, self.Oh, *kernel_size), requires_grad=True)

        # Initialize weights
        xavier_uniform(self.W)
        xavier_uniform(self.U)
        xavier_uniform(self.C)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = []
        for t in range(x.size(1)):
            xt = x.select(dim=1, index=t)
            h = GRU_RCN_CPP_Function.apply(xt, self.W, self.U, self.C, h)
            output.append(h)

        return torch.stack(output, dim=1), h
# endregion


# region Reference implementations
class Ref_GRU_RCN_Vanilla_Impl(nn.Module):
    """
    Vanilla implementation of the GRU-RCN layer using the python frontend.
    """
    def __init__(self, in_channels: int, nb_kernels: int, kernel_size: Tuple[int, int], batch_first: bool = True):
        super(Ref_GRU_RCN_Vanilla_Impl, self).__init__()
        self.Ox = in_channels
        self.Oh = nb_kernels

        if not batch_first:
            raise NotImplementedError("Batch dimensions needs to be first at the moment!")

        # Update gate
        layer_params = dict(out_channels=self.Oh, kernel_size=kernel_size, padding="same", bias=False)
        self.convWz = nn.Conv2d(in_channels=self.Ox, *layer_params)
        self.convUz = nn.Conv2d(in_channels=self.Oh, *layer_params)

        # Reset gate
        self.convWr = nn.Conv2d(in_channels=self.Ox, *layer_params)
        self.convUr = nn.Conv2d(in_channels=self.Oh, *layer_params)

        # Candidate activation
        self.convWh = nn.Conv2d(in_channels=self.Ox, *layer_params)
        self.convUh = nn.Conv2d(in_channels=self.Oh, *layer_params)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = []
        for t in range(x.size(1)):
            x_t = x.select(1, t)

            # Update gate
            z = F.sigmoid(self.convWz(x_t) + self.convUz(h))

            # Reset gate
            r = torch.sigmoid(self.convWr(x_t) + self.convUr(h))

            # Candidate activation
            h_tilde = torch.tanh(self.convWh(x_t) + self.convUh(r * h))

            # Cell output
            h = (1 - z) * h + z * h_tilde
            output.append(h[:, None, :, :, :])

        output = torch.concatenate(output, dim=1)
        return output, h


class Ref_GRU_RCN_Optimized_Impl(nn.Module):
    """
    Restructured implementation of the GRU-RCN layer by reducing the number of convolution calls.
    """
    def __init__(self, in_channels: int, nb_kernels: int, kernel_size: Tuple[int, int], batch_first: bool = True):
        super(Ref_GRU_RCN_Optimized_Impl, self).__init__()
        self.Ox = in_channels
        self.Oh = nb_kernels

        if not batch_first:
            raise NotImplementedError("Batch dimensions needs to be first at the moment!")

        self.W = nn.Parameter(torch.empty(3 * self.Oh, self.Ox, *kernel_size), requires_grad=True)
        self.U = nn.Parameter(torch.empty(2 * self.Oh, self.Oh, *kernel_size), requires_grad=True)
        self.C = nn.Parameter(torch.empty(1 * self.Oh, self.Oh, *kernel_size), requires_grad=True)

        # Initialize weights
        xavier_uniform(self.W)
        xavier_uniform(self.U)
        xavier_uniform(self.C)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = []

        # Iterate over each time step
        for t in range(x.size(1)):
            xt = x.select(1, t)

            gate_weights_w = F.conv2d(xt, self.W, padding="same")
            gate_weights_u = F.conv2d(h, self.U, padding="same")

            cWz, cWr, cWc = gate_weights_w.chunk(3, dim=1)
            cUz, cUr = gate_weights_u.chunk(2, dim=1)

            # Update gate
            z = F.sigmoid(cWz + cUz)

            # Reset gate
            r = F.sigmoid(cWr + cUr)

            # Candidate activation
            c = torch.tanh(cWc + F.conv2d(r * h, self.C, padding="same"))

            # Cell output
            h = (1 - z) * h + z * c
            output.append(h[:, None, :, :, :])

        output = torch.concatenate(output, dim=1)
        return output, h
# endregion
