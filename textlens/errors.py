"""
textlens.errors
───────────────
Structured exception hierarchy.

Every TextLens exception derives from :class:`TextLensError` and carries a
stable machine-readable ``code`` plus an optional human ``hint`` that tells
the user what to do next.  The server maps these codes to HTTP statuses and
the CLI prints the hint under the error message.

Exceptions that replace a standard-library error also inherit from it
(``InputNotFoundError`` is a ``FileNotFoundError``, ``UnsupportedInputError``
is a ``ValueError``) so pre-2.0 ``except`` clauses keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class TextLensError(Exception):
    """Base exception for all TextLens errors."""

    code: str = "textlens_error"
    http_status: int = 500

    def __init__(self, message: str = "", *, hint: Optional[str] = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: Dict[str, Any] = details

    def __str__(self) -> str:
        base = self.message or super().__str__()
        return f"{base}\nHint: {self.hint}" if self.hint else base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": {k: v for k, v in self.details.items() if _jsonable(v)},
        }


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


# ── Configuration ────────────────────────────────────────────────────────────


class ConfigurationError(TextLensError):
    code = "configuration_error"
    http_status = 400


# ── Inputs ───────────────────────────────────────────────────────────────────


class InputError(TextLensError):
    code = "input_error"
    http_status = 400


class InputNotFoundError(InputError, FileNotFoundError):
    code = "input_not_found"
    http_status = 404


class UnsupportedInputError(InputError, ValueError):
    code = "unsupported_input"
    http_status = 415


class InputTooLargeError(InputError):
    code = "input_too_large"
    http_status = 413


class DocumentError(InputError):
    """A document could be read but is malformed, encrypted or empty."""

    code = "document_error"
    http_status = 422


class EncryptedDocumentError(DocumentError):
    code = "encrypted_document"


# ── Models ───────────────────────────────────────────────────────────────────


class ModelError(TextLensError):
    code = "model_error"


class UnknownModelError(ModelError):
    """Raised when an unregistered model ID is requested."""

    code = "unknown_model"
    http_status = 404

    def __init__(self, model_id: str, supported_ids: List[str]) -> None:
        self.model_id = model_id
        self.supported_ids = list(supported_ids)
        supported_str = "\n  • ".join(self.supported_ids)
        super().__init__(
            f'\nUnknown model: "{model_id}"\n\nSupported models are:\n  • {supported_str}\n',
            hint="Run `textlens models list` to see every registered model.",
        )
        self.details = {"model_id": model_id, "supported": self.supported_ids}

    def __str__(self) -> str:  # keep the historical message format
        return self.message


class ModelNotInstalledError(ModelError):
    """Raised when a model is required locally but is not in the cache."""

    code = "model_not_installed"
    http_status = 409

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f'Model "{model_id}" is not installed. Run: textlens models install {model_id}',
        )
        self.details = {"model_id": model_id}

    def __str__(self) -> str:
        return self.message


class DownloadError(ModelError):
    """Raised when a model download fails."""

    code = "download_failed"
    http_status = 502

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        super().__init__(f'Failed to download "{model_id}": {reason}')
        self.details = {"model_id": model_id}

    def __str__(self) -> str:
        return self.message


class IntegrityError(DownloadError):
    """A downloaded artifact did not match its pinned SHA-256 checksum."""

    code = "integrity_error"


class OfflineError(ModelError):
    """A download was required while offline mode is enabled."""

    code = "offline"
    http_status = 409


# ── Backends / runtime ───────────────────────────────────────────────────────


class BackendError(TextLensError):
    code = "backend_error"


class BackendUnavailableError(BackendError):
    """An optional runtime (torch, transformers, onnxruntime, …) is missing."""

    code = "backend_unavailable"
    http_status = 501


class BackendLoadError(BackendError):
    code = "backend_load_failed"


class InferenceError(BackendError):
    code = "inference_failed"


class HardwareInspectionError(TextLensError):
    """Raised when hardware inspection cannot complete."""

    code = "hardware_inspection_failed"


class RoutingError(TextLensError):
    """No model satisfies the requested profile, task and constraints."""

    code = "no_suitable_model"
    http_status = 422


# ── Jobs / serving ───────────────────────────────────────────────────────────


class JobError(TextLensError):
    code = "job_error"


class JobNotFoundError(JobError):
    code = "job_not_found"
    http_status = 404


class CancelledError(JobError):
    code = "cancelled"
    http_status = 409


class CapacityError(TextLensError):
    """Backpressure: the processing queue is full."""

    code = "over_capacity"
    http_status = 503


# ── ANPR ─────────────────────────────────────────────────────────────────────


class ANPRError(TextLensError):
    code = "anpr_error"


__all__ = [name for name in dir() if name.endswith("Error")]
