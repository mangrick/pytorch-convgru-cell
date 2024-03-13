#include <torch/extension.h>
#include <vector>

/**
 * Compute local gradient of the sigmoid function
 */
at::Tensor d_sigmoid(at::Tensor z)
{
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

/**
 * Compute local gradient of the tanh function
 */
at::Tensor d_tanh(at::Tensor z)
{
    return 1 - z.tanh().pow(2);
}

/**
 * Wrapper function for the convolution operation (will use optimized function from MKLDNN when on CPU)
 */
inline at::Tensor conv2d(at::Tensor & x, at::Tensor & w)
{
    return at::mkldnn_convolution(x, w, {}, 1, 1, 1, 1);
}

/**
 * Wrapper function for the backward pass of the convolution operation
 */
std::vector<at::Tensor> d_conv2d(at::Tensor x, at::Tensor w, at::Tensor grad)
{
    auto result = at::convolution_backward(grad, x, w, {}, 1, 1, 1, false, 0, 1, {true, true, false});
    return { std::get<0>(result), std::get<1>(result) };
}

/**
 * Forward pass of the GRU-RCN cell
 */
std::vector<at::Tensor> gru_rcn_forward(at::Tensor x,
                                        at::Tensor W,
                                        at::Tensor U,
                                        at::Tensor C,
                                        at::Tensor h)
{
    torch::NoGradGuard no_grad;

    auto gate_weights_w = conv2d(x, W);
    auto gate_weights_u = conv2d(h, U);

    auto gw_w = gate_weights_w.chunk(3, 1);
    auto conv_Wz = gw_w.at(0);
    auto conv_Wr = gw_w.at(1);
    auto conv_Wc = gw_w.at(2);

    auto gw_u = gate_weights_u.chunk(2, 1);
    auto conv_Uz = gw_u.at(0);
    auto conv_Ur = gw_u.at(1);

    // Update gate
    auto gw_z = conv_Wz + conv_Uz;
    auto z = torch::sigmoid(gw_z);

    // Reset gate
    auto gw_r = conv_Wr + conv_Ur;
    auto r = torch::sigmoid(gw_r);

    // Candidate activation
    auto rh = r * h;
    auto conv_Uc = conv2d(rh, C);
    auto gw_c = conv_Wc + conv_Uc;
    auto c = torch::tanh(gw_c);

    // Cell output
    h = (1 - z) * h + z * c;
    return { h, gw_z, gw_r, gw_c, z, r, rh, c };
}

/**
 * Backward pass of the GRU-RCN cell
 */
std::vector<at::Tensor> gru_rcn_backward(at::Tensor grad,
                                         at::Tensor W,
                                         at::Tensor U,
                                         at::Tensor Uc,
                                         at::Tensor gw_z,
                                         at::Tensor gw_r,
                                         at::Tensor gw_c,
                                         at::Tensor z,
                                         at::Tensor r,
                                         at::Tensor rh,
                                         at::Tensor c,
                                         at::Tensor xt,
                                         at::Tensor h)
{
    torch::NoGradGuard no_grad;

    // Get kernels
    auto w_kernels = W.chunk(3, 0);
    auto u_kernels = U.chunk(2, 0);

    auto Wz = w_kernels.at(0);
    auto Wr = w_kernels.at(1);
    auto Wc = w_kernels.at(2);

    auto Uz = u_kernels.at(0);
    auto Ur = u_kernels.at(1);

    // partial derivatives regarding H
    auto dH_dz = c - h;
    auto dH_dh = 1 - z;
    auto dH_dc = z;

    // partial derivatives regarding z (w.r.t. loss gradient)
    std::vector<at::Tensor> conv_zw = d_conv2d(xt, Wz, d_sigmoid(gw_z) * dH_dz * grad);
    auto dz_dx = conv_zw.at(0);
    auto d_wz  = conv_zw.at(1);

    std::vector<at::Tensor> conv_zu = d_conv2d(h,  Uz, d_sigmoid(gw_z) * dH_dz * grad);
    auto dz_dh = conv_zu.at(0);
    auto d_uz  = conv_zu.at(1);

    // partial derivatives regarding r (w.r.t. loss gradient)
    std::vector<at::Tensor> conv_rh = d_conv2d(rh, Uc, d_tanh(gw_c) * z * grad);
    auto d_rh = conv_rh.at(0);

    std::vector<at::Tensor> conv_rw = d_conv2d(xt, Wr, d_sigmoid(gw_r) * h * d_rh);
    auto dr_dx = conv_rw.at(0);
    auto d_wr  = conv_rw.at(1);

    std::vector<at::Tensor> conv_ru = d_conv2d(h, Ur, d_sigmoid(gw_r) * h * d_rh);
    auto dr_dh = conv_ru.at(0);
    auto d_ur  = conv_ru.at(1);

    // partial derivatives regarding candidate function (w.r.t. loss gradient)
    std::vector<at::Tensor> conv_cw = d_conv2d(xt, Wc, d_tanh(gw_c) * dH_dc * grad);
    auto dc_dx = conv_cw.at(0);
    auto d_wc  = conv_cw.at(1);

    std::vector<at::Tensor> conv_cu = d_conv2d(rh, Uc, d_tanh(gw_c) * dH_dc * grad);
    auto dc_dh = conv_cu.at(0) * r;
    auto d_uc  = conv_cu.at(1);

    // Compute dx and dh
    auto dx = dz_dx + dr_dx + dc_dx;
    auto dh = dz_dh + dr_dh + dc_dh + dH_dh * grad;

    return {dx, dh, d_wz, d_wr, d_wc, d_uz, d_ur, d_uc };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gru_rcn_forward, "GRU RCN forward");
  m.def("backward", &gru_rcn_backward, "GRU RCN backward");
}
