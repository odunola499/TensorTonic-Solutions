import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    results = []
    change = 0.0

    values = np.asarray(values, dtype = np.float64)
    gradients = np.asarray(gradients, dtype = np.float64)

    next_values = values - gradients * learning_rate
    pred_obj_change = gradients * (next_values - values)
    pred_obj_change = np.sum(pred_obj_change)

    return next_values.tolist(), float(pred_obj_change)
