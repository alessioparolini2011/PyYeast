"""
errors.py
Authors: Alessio Parolini
This file contains the customized Exception classes for the package.
In this way, the user can receive a more precise analysis of the errors returned,
understanding better what causes the error.
"""

__all__ = ["YeastException",
           "YeastTypeError",
           "YeastValueError",
           "YeastRuntimeError"
           ]


class YeastException(Exception):
    """
    The default exception for the class, so the user can recognize immediately where the error comes from
    """

    def __init__(self, message: str = ""):
        """
        This is a customized __init__, to format the exception with a default message
        :param message: the specific phrase
        """
        full = f"PyYeast returned an error: {message} - See docs at: https://github.com/alessioparolini2011/PyYeast#readme"
        super().__init__(full)  # show the error in the terminal


# SPECIFIC EXCEPTIONS

class YeastTypeError(YeastException, TypeError):
    """Type error for PyYeast"""
    pass

class YeastValueError(YeastException, ValueError):
    """Value error for PyYeast"""
    pass

class YeastRuntimeError(YeastException, RuntimeError):
    """Runtime error for PyYeast"""
    pass