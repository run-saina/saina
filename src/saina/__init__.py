"""Saina HTTP client and optional local model integrations."""
from .client import (Saina, SainaHelm, ResponseMetadata, Answers, DEFAULT_BASE_URL, new_idempotency_key)
from .errors import (
    SainaError, SainaHelmError, SainaConnectionError, HTTPError, AuthenticationError,
    PermissionDeniedError, NotFoundError, InvalidRequestError, RateLimitError, IdempotencyError,
    ServerError, ModelBusyError, TransportError, ProtocolError, SpoolFullError,
    TraceStorageError, InvalidApiKey, KeyRevoked, FreshAuthRequired, PermissionDenied,
    AccountSuspended, NotFound, InvalidRequest, RateLimited, InsufficientCredits,
    IdempotencyConflict, RequestInProgress, IdempotencyResultExpired, IdempotencyUnverifiable, InferenceFailed,
    AccountingUnavailable, Overloaded, InferenceUnavailable, ERROR_CLASSES
)
from .answers import (YesNoAnswer, SingleChoiceAnswer, RatingAnswer, MultiChoiceAnswer,
                      Selections, YesNoSelection, SingleChoiceSelection, RatingSelection,
                      MultiChoiceSelection, SelectionReason)
from .traces import Traces, TraceFlushResult, TraceClient, DecisionTrace, TraceDeliveryError
__version__ = '1.0.0rc1'

def register_transformers():
    from .transformers import register
    register()


def load_model(checkpoint, **kwargs):
    from .transformers import HelmModel
    return HelmModel.from_pretrained(checkpoint, **kwargs)
