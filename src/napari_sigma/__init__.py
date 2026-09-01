from ._version import __version__

__all__ = ["SIGMAWidget", "__version__"]


def __getattr__(name):
    if name == "SIGMAWidget":
        from ._widget import SIGMAWidget

        return SIGMAWidget
    raise AttributeError(name)
