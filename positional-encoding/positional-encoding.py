import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(seq_len).reshape(-1, 1)
    dims = np.arange(0, d_model, 2)
    output = np.zeros((seq_len, d_model))

    angles = positions / base ** (dims / d_model)

    output[:, 0::2] = np.sin(angles)
    output[:, 1::2] = np.cos(angles[:, :(output[:,1::2].shape[-1])])
    return output