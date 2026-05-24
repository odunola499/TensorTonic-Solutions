import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    embed_dim = K.shape[-1]
    attn_scores = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(embed_dim), dim = -1)
    output = attn_scores @ V
    return output
    