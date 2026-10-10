"""Saina HTTP client and optional local model integrations."""
from .client import (Saina, SainaError, SainaConnectionError, ResponseMetadata, ERROR_CLASSES,
                     DEFAULT_BASE_URL, new_idempotency_key)
__version__ = '0.3.0'

def register_transformers():
    from .transformers import register
    register()


def load_model(checkpoint, **kwargs):
    from .transformers import HelmModel
    return HelmModel.from_pretrained(checkpoint, **kwargs)
