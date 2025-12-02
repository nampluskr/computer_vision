"""
Backend module for seamless NumPy/CuPy switching
Usage:
    from backend import backend
    xp = backend.xp
"""

import numpy as _np

try:
    import cupy as _cp
    GPU_AVAILABLE = True
except ImportError:
    _cp = None
    GPU_AVAILABLE = False


class Backend:
    """Backend wrapper for NumPy/CuPy compatibility"""
    
    def __init__(self, use_gpu=True):
        """
        Args:
            use_gpu (bool): Use GPU if available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = _cp if self.use_gpu else _np
        
    def asarray(self, x, dtype=None):
        """Convert array to appropriate backend"""
        if self.use_gpu:
            return self.xp.asarray(x, dtype=dtype)
        return _np.asarray(x, dtype=dtype)
    
    def to_cpu(self, x):
        """Move array to CPU (NumPy)"""
        if self.use_gpu and hasattr(self.xp, "asnumpy"):
            return self.xp.asnumpy(x)
        return x
    
    def to_gpu(self, x):
        """Move array to GPU (CuPy)"""
        if self.use_gpu:
            return self.xp.asarray(x)
        return x


# Global backend instance
backend = Backend(use_gpu=True)

# Convenience exports
xp = backend.xp
asarray = backend.asarray
to_cpu = backend.to_cpu
to_gpu = backend.to_gpu