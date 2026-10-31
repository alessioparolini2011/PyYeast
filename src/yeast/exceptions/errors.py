"""
errors.py
Authors: Alessio Parolini
This file contains the customized Exception classes for the package.
"""

__all__ = ["SwellError", "XLayersError", "LayersTypeError", "LayersValueError", "XOptimizerError", "OptimizerValueError", "OptimizerTypeError"]

class SwellError(Exception):

    """
    The default package exception
    """

    pass

# --- LAYERS ERRORS ---

class XLayersError(SwellError):

    """
    The default error for layers resources
    """

    pass

class LayersTypeError(XLayersError, TypeError):

    """
    The exception for wrong types in layers
    """

    pass

class LayersValueError(XLayersError, ValueError):

    """
    The exception for wrong values in layers
    """

    pass

class XOptimizerError(SwellError):

    """
    The base exception for optimizers
    """

class OptimizerTypeError(XOptimizerError, TypeError):

    """
    The exception for wrong types in optimizers
    """

    pass

class OptimizerValueError(XOptimizerError, ValueError):

    """
    The exception for wrong values in optimizers
    """
