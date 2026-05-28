import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    
    m_t = np.multiply(beta1, m) + np.multiply((1-beta1), grad)
    v_t = np.multiply(beta2, v) + np.multiply((1-beta2), np.square(grad))

    m_t_hat = m_t / (1-(beta1**t))
    v_t_hat = v_t / (1-(beta2**t))

    effective_gradient = m_t_hat / (np.sqrt(v_t_hat) + eps)

    param_new = param - (lr * effective_gradient)
    return (param_new, m_t, v_t)