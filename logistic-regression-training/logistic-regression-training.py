import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def model(X, W, b):
    z = (X @ W) + b
    return _sigmoid(z)

def _loss(p, y):
    N = p.shape[0]
    result = (y*np.log(p)) + ((1-y) * np.log(1-p))
    return -np.mean(result)
    
def _gradients(X, p, y):
    N = X.shape[0]
    grad_w = (X.T @ (p-y)) / N
    grad_b = np.mean(p-y)
    return grad_w, grad_b
    
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    D = X.shape[-1]
    W = np.zeros((D,))
    b = 0.0
    for step in range(steps):
        p = model(X, W, b)
        loss = _loss(p, y)
        grad_w, grad_b = _gradients(X, p, y)

        W -= lr * grad_w
        b -= lr * grad_b

    return (W, b)
        
    