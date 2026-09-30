def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    a, b, c, h = float(a), float(b), float(c), float(h)
    sum_0 = (a*b) + c

    d_a = ((a + h) * b) + c
    d_b = (a * (b + h)) + c
    d_c = (a * b) + (c + h)

    partial_a = float((d_a - sum_0) / h)
    partial_b = float((d_b - sum_0) / h)
    partial_c = float((d_c - sum_0) / h)

    return float(sum_0), partial_a, partial_b, partial_c
