# __init__.py
__all__ = []
__version__ = "0.1.0"

def main(*args, **kwargs):
    from .main import main as _main  # imported only when called
    return _main(*args, **kwargs)