import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    embed_dim = Q.shape[-1]
    seq_len = Q.shape[-2]
    head_dim = embed_dim // num_heads

    Q = np.matmul(Q, W_q)
    K = np.matmul(K, W_k)
    V = np.matmul(V, W_v)
    
    
    Q = Q.reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(-1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_scores = softmax(np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim))
    attention = np.matmul(attn_scores, V)

    output = attention.transpose(0, 2, 1, 3).reshape(-1, seq_len, embed_dim)
    output = np.matmul(output, W_o)

    return output
    
    

    
    
    