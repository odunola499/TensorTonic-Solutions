import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max([len(i) for i in seqs])

    padded_seqs = []
    for seq in seqs:
        seq = seq[:max_len]
        length = len(seq)
        if length == max_len:
            padded_seqs.append(seq)
        else:
            extras = max_len - length
            pads = [pad_value] * extras
            seq += pads
            padded_seqs.append(seq)

    final_seq = np.stack(padded_seqs)
    return final_seq
        