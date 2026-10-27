"""SDK failure categories and server code-specific exceptions."""

class InputTooLong(ValueError):
    """The full input exceeds the configured context window."""


class ModelNotServed(Exception):
    """The request names a specific model version this endpoint does not serve."""


class SainaError(RuntimeError):
    """Base class for SDK failures."""
    code = None

    def __init__(self, message, status=None, *, code=None, request_id=None, admitted=None, state=None,
                 retry=None, retry_after=None, suspension=None, raw=None, fields=None):
        super().__init__(message)
        self.status = status
        self.code = code or type(self).code
        self.request_id, self.admitted, self.state = request_id, admitted, state
        self.retry, self.retry_after, self.suspension = retry, retry_after, suspension
        self.raw = raw
        self.fields = dict(fields or {})


class HTTPError(SainaError):
    """An unsuccessful HTTP response."""


class AuthenticationError(HTTPError):
    pass


class PermissionDeniedError(HTTPError):
    pass


class NotFoundError(HTTPError):
    pass


class InvalidRequestError(HTTPError):
    pass


class RateLimitError(HTTPError):
    pass


class IdempotencyError(HTTPError):
    pass


class ServerError(HTTPError):
    pass


class ModelBusyError(ServerError):
    pass


class TransportError(SainaError):
    def __init__(self, message, *args, timed_out=False, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.timed_out = timed_out


class ProtocolError(SainaError):
    pass


class SpoolFullError(SainaError):
    pass


class TraceStorageError(SainaError):
    pass


SainaHelmError = SainaError
SainaConnectionError = TransportError


class InvalidApiKey(AuthenticationError):
    code = 'invalid_api_key'


class KeyRevoked(AuthenticationError):
    code = 'key_revoked'


class FreshAuthRequired(AuthenticationError):
    code = 'fresh_auth_required'


class PermissionDenied(PermissionDeniedError):
    code = 'permission_denied'


class AccountSuspended(PermissionDeniedError):
    code = 'account_suspended'


class NotFound(NotFoundError):
    code = 'not_found'


class InvalidRequest(InvalidRequestError):
    code = 'invalid_request'


class RateLimited(RateLimitError):
    code = 'rate_limited'


class InsufficientCredits(HTTPError):
    code = 'insufficient_credits'


class IdempotencyConflict(IdempotencyError):
    code = 'idempotency_conflict'


class RequestInProgress(IdempotencyError):
    code = 'request_in_progress'


class IdempotencyResultExpired(IdempotencyError):
    code = 'idempotency_result_expired'


class IdempotencyUnverifiable(IdempotencyError):
    code = 'idempotency_unverifiable'


class InferenceFailed(ServerError):
    code = 'inference_failed'


class AccountingUnavailable(ServerError):
    code = 'accounting_unavailable'


class Overloaded(ModelBusyError):
    code = 'overloaded'


class InferenceUnavailable(ModelBusyError):
    code = 'inference_unavailable'


ERROR_CLASSES = {cls.code: cls for cls in (
    InvalidApiKey, KeyRevoked, FreshAuthRequired, PermissionDenied, AccountSuspended,
    NotFound, InvalidRequest, RateLimited, InsufficientCredits, IdempotencyConflict,
    RequestInProgress, IdempotencyResultExpired, IdempotencyUnverifiable, InferenceFailed,
    AccountingUnavailable, Overloaded, InferenceUnavailable,
)}
