import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    
    m_t = np.multiply(beta1, m) + np.multiply((1-beta1), grad)
    v_t = np.multiply(beta2, v) + np.multiply((1-beta2), np.square(grad))


    decay = lr * np.multiply(w, weight_decay)
    grad_step = lr * (m_t / (np.sqrt(v_t) + eps))
    step =  decay + grad_step
    w_t = w - step
    return (w_t, m_t, v_t)