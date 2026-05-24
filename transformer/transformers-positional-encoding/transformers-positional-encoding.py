import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    positions = np.arange(seq_length).reshape(-1,1)
    dims = np.arange(0, d_model, 2).reshape(1, -1)
    angles = positions * np.exp(dims * (-np.log(10000.0) / d_model))
    
    output = np.zeros((seq_length, d_model))

    output[:,0::2] = np.sin(angles)
    output[:,1::2] = np.cos(angles)
    return output

    
    
    