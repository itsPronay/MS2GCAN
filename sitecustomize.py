import numpy as np

# Compatibility for legacy NumPy aliases used by older scientific packages.
# NumPy 2.x removed aliases like np.int, np.float, and np.bool.
for name, value in {
    'int': int,
    'float': float,
    'complex': complex,
    'bool': bool,
    'object': object,
}.items():
    if not hasattr(np, name):
        setattr(np, name, value)
