def compute_min_max(data, num_rows, num_cols):
    min_list = [float('inf')]  * num_cols
    max_list = [float('-inf')] * num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            if data[row][col] >= max_list[col]:
                max_list[col] = data[row][col]
            if data[row][col] <= min_list[col]:
                min_list[col] = data[row][col]

    range_list = [i-j for i,j in zip(max_list, min_list)]
    return min_list, max_list, range_list

def _scale(point, min_val, max_val):
    return (point - min_val) / (max_val - min_val)

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    num_rows = len(data)
    num_cols = len(data[0])

    result = [[0]*num_cols for _ in range(num_rows)]
    min_list, max_list, range_list = compute_min_max(data, num_rows, num_cols)

    for row in range(num_rows):
        for col in range(num_cols):
            min_val = min_list[col]
            max_val = max_list[col]
            range_val = range_list[col]

            if range_val == 0:
                result[row][col] = 0
            else:
                result[row][col] = _scale(data[row][col],min_val, max_val)

    return result
                
    
    