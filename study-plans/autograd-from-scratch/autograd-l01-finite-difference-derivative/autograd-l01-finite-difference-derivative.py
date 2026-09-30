import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    f_x = 0
    f_x_h = 0
    for index, coeff in enumerate(coefficients):
        res_x = coeff * (x ** index)
        res_x_h = coeff * ((x + h)** index)

        f_x += res_x
        f_x_h += res_x_h

    slope = (f_x_h - f_x) / h
    return (float(f_x), float(f_x_h), float(slope))
