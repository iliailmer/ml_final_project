"""Swish activation function from here: https://arxiv.org/abs/1710.05941 ."""
import torch


@torch.jit.script
def swish(x):
    """Compute Baseline Swish."""
    return x*x.sigmoid()


@torch.jit.script
def swish_back(x, grad_output):
    """Compute grad of Swish."""
    sigmoid = x.sigmoid()
    g = sigmoid*(1. + x*(1.-sigmoid))
    return grad_output*g  # chain rule


class SwishFunction(torch.autograd.Function):
    """Autograd version of swish with PyTorch."""

    @staticmethod
    def forward(ctx, x):
        """Perform Forward Pass."""
        ctx.save_for_backward(x)
        return swish(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Perform Backward Pass."""
        return swish_back(x=ctx.saved_tensors[0],
                          grad_output=grad_output)


class Swish(torch.nn.Module):
    """Swish Activation Function - PyTorch CUDA Version."""

    def forward(self, inp):
        """Perform Backward Pass."""
        return SwishFunction.apply(inp)
