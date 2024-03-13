# ConvGRU Cell for Pytorch

A reimplementation of the GRU-RCN cell from the paper "[Delving deeper into convolutional networks for learning video representations](https://arxiv.org/abs/1511.06432)" by Ballas et al. (ICLR, 2016) as a C++ extension for the PyTorch deep learning framework.

## Installation

The recommended steps for installing are to either clone the repository and install the package into the local environment using pip or by pointing the GitHub repository directly. In case a specific PyTorch version is required, clone the repository and build the shared library directly with setup.py.
```bash
git clone git@github.com:mangrick/pytorch-convgru-cell.git
pip install pytorch-convgru-cell

# Alternative
cd pytorch-convgru-cell
python setup.py build_ext --inplace
```

## Usage

The GRU-RCN cell requires input tensors with a dimension of 5, representing the batch size, number of time steps, channels, and two dimensions for the spatial sizes. The following snipped provides an example:

```python
import torch
from ballas2016 import GRU_RCN

N1 = 8   # spatial dim 1
N2 = 8   # spatial dim 2
Ox = 5   # channels (number of feature maps of the preceding convolutional layer)
Oh = 32  # Dimensionality of the GRU-RCN hidden representation

L = 100  # Number of time steps in the sequence
B = 16   # batch size

input = torch.rand((B, L, Ox, N1, N2))
state = torch.rand((Oh, N1, N2))

rnn = GRU_RCN(in_channels=Ox, nb_kernels=Oh, kernel_size=(3, 3))
print(f"#Parameters: {sum(p.numel() for p in rnn.parameters() if p.requires_grad)}")

output, new_state = rnn(input, state)
```

## Reference implementation

The following code snippet corresponds to a vanilla reference implementation compared to the C++ version.

```python
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
```
## Performance comparison

The C++ extension was compared against a reference GRU-RCN layer implemented in PyTorch using the Python frontend. This reference layer was designed analogous to the function definitions in the paper. Since the convolution operation is one of the most expensive calls in the GRU-RCN layer, this is not an optimal design and the computational cost can further be reduced by combining kernel filters and inputs. 

Overall, the C++ extension outperforms the reference implementation by around 40% in the forward pass (27% where achieved by combining convolutional inputs and having only 2 calls to the convolution function instead of 6). The backward pass only mildly achieved a better performance (around 6%).

The test setup for the performance comparison is given below:
```bash
SETUP=$(cat <<-END
import torch
from ballas2016 import Ref_GRU_RCN_Vanilla_Impl as GRU_RCN_REF
from ballas2016 import GRU_RCN as GRU_RCN_CPP

x = torch.rand((16, 10, 5, 8, 8))
h = torch.rand((16, 32, 8, 8))

rnn_ref = GRU_RCN_REF(in_channels=5, nb_kernels=32, kernel_size=(3, 3))
rnn_cpp = GRU_RCN_CPP(in_channels=5, nb_kernels=32, kernel_size=(3, 3))

with torch.no_grad():
    y_cpp, h_cpp = rnn_cpp(x, h)
    y_ref, h_ref = rnn_ref(x, h)
y_cpp.requires_grad = True
y_ref.requires_grad = True
grad = torch.rand((16, 10, 32, 8, 8))
END
)

python -m timeit -n 100 -r 100 -s $SETUP "y1, h1 = rnn_cpp(x, h)"
python -m timeit -n 100 -r 100 -s $SETUP "h2, h2 = rnn_ref(x, h)"

python -m timeit -n 100 -r 100 -s $SETUP "y_cpp.backward(grad)"
python -m timeit -n 100 -r 100 -s $SETUP "y_ref.backward(grad)"
```

When run on tensors in CPU memory, the forward pass will rely on an optimized implementation from the MKLDNN library which is usually included in the PyTorch build.

## Technical validity

The C++ extension was compared against the vanilla reference implementation for technical correctness.
```python
import torch
from ballas2016 import Ref_GRU_RCN_Vanilla_Impl as GRU_RCN_REF
from ballas2016 import GRU_RCN as GRU_RCN_CPP

# Use float64 by default (see Limitations)
torch.set_default_dtype(torch.float64)

N1 = 8  # spatial size 1
N2 = 8  # spatial size 2
Ox = 5  # channels (or number of feature maps)
Oh = 32  # Dimensionality of the GRU-RCN hidden representation
B = 16  # batch size
L = 5  # Number of time steps

# Prepare input examples
x1 = torch.rand((B, L, Ox, N1, N2), requires_grad=True)
h1 = torch.rand((B, Oh, N1, N2), requires_grad=True)

x2 = x1.clone().detach().requires_grad_(True)
h2 = h1.clone().detach().requires_grad_(True)

# Set up network layers
rnn_ref = GRU_RCN_REF(in_channels=Ox, nb_kernels=Oh, kernel_size=(3, 3))
rnn_cpp = GRU_RCN_CPP(in_channels=Ox, nb_kernels=Oh, kernel_size=(3, 3))

# Ensure that the internal weights are matching
rnn_cpp.W.data = torch.cat([rnn_ref.convWz.weight, rnn_ref.convWr.weight, rnn_ref.convWh.weight], dim=0)
rnn_cpp.U.data = torch.cat([rnn_ref.convUz.weight, rnn_ref.convUr.weight, ], dim=0)
rnn_cpp.C.data = rnn_ref.convUh.weight.copy()

# Check forward pass
y1, new_h1 = rnn_ref(x1, h1)
y2, new_h2 = rnn_cpp(x2, h2)

print(f"x: {torch.all(torch.isclose(y1, y2))}")
print(f"h: {torch.all(torch.isclose(new_h1, new_h2))}")

# Check backward pass
grad = torch.rand(x1.shape)
y1.backward(grad)
y2.backward(grad)

print(f"dx: {torch.all(torch.isclose(x1.grad, x2.grad))}")
print(f"dh: {torch.all(torch.isclose(h1.grad, h2.grad))}")

# Check if weight gradients match
x_rnn_dW = torch.cat([rnn_ref.convWz.weight.grad, rnn_ref.convWr.weight.grad, rnn_ref.convWh.weight.grad], dim=0)
x_rnn_dU = torch.cat([rnn_ref.convUz.weight.grad, rnn_ref.convUr.weight.grad], dim=0)

print(f"dW: {torch.all(torch.isclose(x_rnn_dW, rnn_cpp.W.grad))}")
print(f"dU: {torch.all(torch.isclose(x_rnn_dU, rnn_cpp.U.grad))}")
print(f"dC: {torch.all(torch.isclose(rnn_ref.convUh.weight.grad, rnn_cpp.C.grad))}")
```

## Limitations
- The current implementation for C++ extension only supports odd kernel dimensions.
- Due to floating-point arithmetics it is recommended to use float64 instead of float32 for numerical more stable results.
- Currently, there is no implementation for CUDA devices
