# implementations from the paper: https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import math
import torch
from torch import nn
from torch.nn import RMSNorm
import torch.nn.functional as F

def activation_quant(x):
  """ Per-token quantization to 8 bits. No grouping is needed for quantization.
  Args:
  x: an activation tensor with shape [n, d]
  Returns:
  y: a quantized activation tensor with shape [n, d]
  """
  scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
  y = (x * scale).round().clamp_(-128, 127) / scale
  return y

def weight_quant(w):
  """ Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
  Args:
  w: a weight tensor with shape [d, k]
  Returns:
  u: a quantized weight with shape [d, k]
  """
  scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
  u = (w * scale).round().clamp_(-1, 1) / scale
  return u
    
class BitLinear(nn.Module):
  """
  This is only for training, and kernel optimization is needed for efficiency.
  """
  def __init__(self, in_features, out_features, bias=False):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features)))
    self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
    self.norm = RMSNorm(in_features)
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      
  def forward(self, x):
    """
    Args:
    x: an input tensor with shape [n, d]
    Returns:
    y: an output tensor with shape [n, d]
    """
    x_norm = self.norm(x)
    # A trick for implementing Straight−Through−Estimator (STE) using detach()
    x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
    w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
    y = F.linear(x_quant, w_quant)
    return y