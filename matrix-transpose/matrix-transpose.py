import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    orig_num_rows = len(A)
    orig_num_cols = len(A[0])

    new_num_rows = orig_num_cols
    new_num_cols = orig_num_rows

    result = [[0] * new_num_cols for _ in range(new_num_rows)]
    for i in range(new_num_rows):
        for j in range(new_num_cols):
            result[i][j] = A[j][i]

    return np.array(result)
