def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    for step in range(steps):
        m = (2 * a * x0) + b
        x0 = x0 - (lr * m)
    return x0