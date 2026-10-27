"""
errors.py
Authors: Alessio Parolini
This file contains the customized Exception classes for the package.
"""

__all__ = ["SwellError", "XLayersError", "LayersTypeError", "LayersValueError"]

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