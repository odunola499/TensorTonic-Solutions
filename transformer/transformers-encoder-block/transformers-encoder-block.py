import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = np.mean(x, axis = -1, keepdims = True)
    variance = np.var(x, axis = -1, keepdims = True)
    denum = np.sqrt(variance + eps)
    output = (gamma* (x - mean) / denum) + beta
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    embed_dim, seq_len = Q.shape[-1], Q.shape[-2]
    head_dim = embed_dim // num_heads
    Q = np.matmul(Q, W_q).reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = np.matmul(K, W_k).reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = np.matmul(V, W_v).reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_scores = softmax(np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim))
    output = np.matmul(attn_scores, V)

    output = output.transpose(0, 2, 1, 3).reshape(-1, seq_len, embed_dim)
    return np.matmul(output, W_o)
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    inner = np.maximum(0, np.matmul(x, W1) + b1)
    return np.matmul(inner, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x_prime = layer_norm(x + multi_head_attention(x,x,x,W_q, W_k, W_v, W_o, num_heads),
                        gamma1, beta1)
    output = layer_norm(x_prime + feed_forward(x_prime, W1,b1, W2, b2),
                       gamma2,beta2)
    return output