class InputTooLong(ValueError):
    """The full input exceeds the configured context window."""


class ModelNotServed(Exception):
    """The request names a specific model version this endpoint does not serve."""
