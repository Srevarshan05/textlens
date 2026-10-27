"""
textlens.inputs.source
──────────────────────
One normalisation point for everything a user can pass to TextLens.

Accepted inputs
    ``str`` / ``pathlib.Path``      local file, or ``http(s)://`` URL
    ``bytes`` / ``bytearray``       raw file content
    file-like objects               anything with ``.read()`` (BytesIO, uploads)
    ``PIL.Image.Image``             in-memory image
    ``numpy.ndarray``               HxW or HxWxC uint8 image (RGB order)

The file type is decided from **magic bytes**, never from the extension, so
misnamed uploads are handled correctly and disguised files are rejected.
Content hashes are computed in streaming fashion and cached, which makes
them cheap to use as result-cache keys.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import logging
import socket
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from textlens.errors import InputError, InputNotFoundError, InputTooLargeError, UnsupportedInputError

logger = logging.getLogger("textlens.inputs")

KIND_PDF = "pdf"
KIND_IMAGE = "image"
KIND_DOCX = "docx"
KIND_PPTX = "pptx"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jp2", ".pbm", ".pgm", ".ppm"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS

_MIME = {
    KIND_PDF: "application/pdf",
    KIND_DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    KIND_PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def sniff(head: bytes, name: str = "") -> tuple[str, str]:
    """Return ``(kind, mime)`` from the first bytes of a file.

    Raises :class:`UnsupportedInputError` for unknown formats.
    """
    if b"%PDF-" in head[:1024]:
        return KIND_PDF, _MIME[KIND_PDF]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return KIND_IMAGE, "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return KIND_IMAGE, "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return KIND_IMAGE, "image/gif"
    if head.startswith(b"BM"):
        return KIND_IMAGE, "image/bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return KIND_IMAGE, "image/tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return KIND_IMAGE, "image/webp"
    if head.startswith(b"\x00\x00\x00\x0cjP  ") or head.startswith(b"\xff\x4f\xff\x51"):
        return KIND_IMAGE, "image/jp2"
    if head[:2] in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"):
        return KIND_IMAGE, "image/x-portable-anymap"
    if head.startswith(b"PK\x03\x04"):
        lower = name.lower()
        if lower.endswith(".docx"):
            return KIND_DOCX, _MIME[KIND_DOCX]
        if lower.endswith(".pptx"):
            return KIND_PPTX, _MIME[KIND_PPTX]
        return "zip", "application/zip"  # resolved by inspecting entries
    raise UnsupportedInputError(
        f"Unsupported file type{f' for {name!r}' if name else ''}.",
        hint="TextLens reads PDF, PNG, JPEG, TIFF, WebP, BMP, GIF, DOCX and PPTX.",
    )


def _zip_kind(data_or_path: Any) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(data_or_path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        raise UnsupportedInputError("Corrupt ZIP/Office file.") from exc
    if any(n.startswith("word/") for n in names):
        return KIND_DOCX, _MIME[KIND_DOCX]
    if any(n.startswith("ppt/") for n in names):
        return KIND_PPTX, _MIME[KIND_PPTX]
    raise UnsupportedInputError("ZIP archives are not supported; pass the documents individually.")


@dataclass
class Source:
    """A normalised input document."""

    kind: str
    mime: str
    name: str
    path: Optional[Path] = None
    data: Optional[bytes] = None
    image: Any = None  # PIL.Image for in-memory image inputs
    origin: str = "file"  # file | bytes | url | image | array
    _sha256: Optional[str] = field(default=None, repr=False)

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        if self.path is not None:
            return self.path.stat().st_size
        return 0

    @property
    def is_pdf(self) -> bool:
        return self.kind == KIND_PDF

    @property
    def is_image(self) -> bool:
        return self.kind == KIND_IMAGE

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is not None:
            return self.path.read_bytes()
        if self.image is not None:
            buf = io.BytesIO()
            self.image.save(buf, format="PNG")
            return buf.getvalue()
        return b""

    def open_binary(self) -> io.BufferedIOBase:
        if self.path is not None and self.data is None:
            return open(self.path, "rb")  # noqa: SIM115 - caller closes
        return io.BytesIO(self.read_bytes())

    @property
    def sha256(self) -> str:
        """Content hash (streamed for files; pixels + mode + size for images)."""
        if self._sha256 is None:
            h = hashlib.sha256()
            if self.image is not None and self.data is None and self.path is None:
                img = self.image
                h.update(f"{img.mode}:{img.size}".encode())
                h.update(img.tobytes())
            elif self.data is not None:
                h.update(self.data)
            elif self.path is not None:
                with open(self.path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            self._sha256 = h.hexdigest()
        return self._sha256


UrlGuard = Callable[[str], None]


def default_url_guard(url: str) -> None:
    """Reject URLs that resolve to private, loopback or link-local addresses.

    Used by the server to prevent SSRF; the library does not apply it by
    default because fetching from a LAN host is a legitimate local use.
    """
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InputError(f"Cannot resolve host {host!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise InputError(f"URL host {host!r} resolves to a non-public address; refusing to fetch.")


def fetch_url(url: str, max_bytes: int, timeout: float = 30.0, guard: Optional[UrlGuard] = None) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsupportedInputError(f"Only http(s) URLs are supported, got {parsed.scheme!r}.")
    if guard is not None:
        guard(url)
    req = urllib.request.Request(url, headers={"User-Agent": "textlens"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme validated above
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise InputTooLargeError(f"Remote file is {int(declared) / 1e6:.1f} MB (limit {max_bytes / 1e6:.0f} MB).")
            buf = io.BytesIO()
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    raise InputTooLargeError(f"Remote file exceeds the {max_bytes / 1e6:.0f} MB limit.")
            return buf.getvalue()
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Could not download {url}: {exc}") from exc


def open_source(
    obj: Any,
    *,
    name: Optional[str] = None,
    allow_urls: Optional[bool] = None,
    max_bytes: Optional[int] = None,
    url_guard: Optional[UrlGuard] = None,
    allowed_root: Optional[Path] = None,
) -> Source:
    """Normalise *obj* into a :class:`Source`.

    Parameters
    ----------
    allow_urls:
        Permit ``http(s)://`` inputs (defaults to ``TEXTLENS_ALLOW_URLS``).
    max_bytes:
        Reject inputs larger than this (defaults to ``TEXTLENS_MAX_FILE_MB``).
    allowed_root:
        If given, local paths must resolve inside this directory (the server
        and MCP tools use this to prevent arbitrary file reads).
    """
    from textlens.config import get_settings

    settings = get_settings()
    if allow_urls is None:
        allow_urls = settings.allow_urls
    if max_bytes is None:
        max_bytes = settings.max_file_mb * 1024 * 1024

    if isinstance(obj, Source):
        return obj

    # PIL image
    try:
        from PIL import Image as PILImage

        if isinstance(obj, PILImage.Image):
            return Source(KIND_IMAGE, "image/x-pil", name or "image", image=obj, origin="image")
    except ImportError:  # pragma: no cover - pillow is a core dependency
        pass

    # numpy array
    if type(obj).__module__ == "numpy" and hasattr(obj, "shape"):
        from textlens.inputs.images import array_to_image

        return Source(KIND_IMAGE, "image/x-array", name or "array", image=array_to_image(obj), origin="array")

    if isinstance(obj, (bytes, bytearray, memoryview)):
        data = bytes(obj)
        return _from_bytes(data, name or "document", max_bytes, origin="bytes")

    if hasattr(obj, "read") and callable(obj.read):
        if hasattr(obj, "seek"):
            try:
                obj.seek(0)
            except Exception:
                pass
        data = obj.read(max_bytes + 1)
        if isinstance(data, str):
            raise UnsupportedInputError("Text streams are not supported; open the file in binary mode ('rb').")
        if len(data) > max_bytes:
            raise InputTooLargeError(f"Input exceeds the {max_bytes / 1e6:.0f} MB limit.")
        return _from_bytes(data, name or getattr(obj, "name", None) or "document", max_bytes, origin="bytes")

    if isinstance(obj, (str, Path)):
        text = str(obj).strip()
        if text.lower().startswith(("http://", "https://")):
            if not allow_urls:
                raise InputError("URL inputs are disabled (TEXTLENS_ALLOW_URLS=0).")
            data = fetch_url(text, max_bytes, guard=url_guard)
            url_name = Path(urllib.parse.urlparse(text).path).name or "download"
            src = _from_bytes(data, name or url_name, max_bytes, origin="url")
            return src
        path = Path(text).expanduser()
        if allowed_root is not None:
            root = Path(allowed_root).resolve()
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if resolved != root and root not in resolved.parents:
                raise InputError(f"Path {text!r} is outside the allowed directory.")
            path = resolved
        if not path.exists():
            raise InputNotFoundError(f"Input file not found: {path}")
        if not path.is_file():
            raise InputError(f"Input is not a file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise InputTooLargeError(f"{path.name} is {size / 1e6:.1f} MB (limit {max_bytes / 1e6:.0f} MB).")
        if size == 0:
            raise InputError(f"{path.name} is empty.")
        with open(path, "rb") as fh:
            head = fh.read(2048)
        kind, mime = sniff(head, path.name)
        if kind == "zip":
            kind, mime = _zip_kind(path)
        return Source(kind, mime, name or path.name, path=path, origin="file")

    raise UnsupportedInputError(
        f"Unsupported input type: {type(obj).__name__}",
        hint="Pass a file path, URL, bytes, a binary file object, a PIL image or a numpy array.",
    )


def _from_bytes(data: bytes, name: str, max_bytes: int, origin: str) -> Source:
    if len(data) > max_bytes:
        raise InputTooLargeError(f"Input is {len(data) / 1e6:.1f} MB (limit {max_bytes / 1e6:.0f} MB).")
    if not data:
        raise InputError("Input is empty.")
    kind, mime = sniff(data[:2048], name)
    if kind == "zip":
        kind, mime = _zip_kind(io.BytesIO(data))
    return Source(kind, mime, name, data=data, origin=origin)
