"""Saina HTTP client and optional local model integrations."""
from .client import Saina, SainaError
__version__ = '0.1.1'

def register_transformers():
    from .transformers import register
    register()


def load_model(checkpoint, **kwargs):
    from .transformers import HelmModel
    return HelmModel.from_pretrained(checkpoint, **kwargs)
