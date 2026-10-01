import numpy as np

def gradient_check_product_chain(
    a: float,
    b: float,
    c: float,
    f: float,
    h: float,
) -> tuple[float, list, list, float]:
    """
    Returns loss, analytic gradients, numerical gradients, and maximum error.
    """
    a, b, c, f, h = map(np.float64, (a, b, c, f, h))
    e = a*b + c
    L = e * f

    # analytical gradients
    d_l_a = f * b
    d_l_b = f * a
    d_l_c = f
    d_l_f = e

    # numerical gradients
    # for a
    num_e = ((a + h) * b) + c
    num_L = num_e * f
    num_d_l_a = (num_L - L) / h

    # for b
    num_e = (a * (b+h)) + c
    num_L = num_e * f
    num_d_l_b = (num_L - L) / h

    # for c
    num_e = (a * b) + (c + h)
    num_L = num_e * f
    num_d_l_c = (num_L - L) / h

    # for f
    num_e = (a * b) + c
    num_L = num_e * (f + h)
    num_d_l_f = (num_L - L) / h 

    analytic = [float(d_l_a), float(d_l_b), float(d_l_c), float(d_l_f)]
    numerical = [
        float(num_d_l_a),
        float(num_d_l_b),
        float(num_d_l_c),
        float(num_d_l_f),
    ]

    max_error = max(
        abs(ana - num)
        for ana, num in zip(analytic, numerical)
    )
    return float(L), analytic, numerical, max_error