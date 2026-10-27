"""
Shared test configuration.

* Every test session runs with an isolated ``TEXTLENS_HOME`` so tests never
  touch the developer's config, result cache or measurements.
* Model weights: tests marked ``@pytest.mark.models`` need the real
  PP-OCRv6 files.  They look in ``TEXTLENS_TEST_MODELS_DIR`` (or the
  default model cache) and are skipped when the model is absent, unless
  ``TEXTLENS_TEST_DOWNLOAD=1`` allows downloading it (~31 MB).
* ``fake_backend`` registers a deterministic in-process OCR engine through
  the public plugin API, so routing, fallback and selective OCR are tested
  without any model weights.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ── isolate TextLens state before any textlens module is imported ───────────
_HOME = Path(tempfile.mkdtemp(prefix="textlens-test-home-"))
os.environ["TEXTLENS_HOME"] = str(_HOME)
os.environ.pop("TEXTLENS_PROFILE", None)
os.environ.pop("TEXTLENS_OFFLINE", None)
os.environ.pop("TEXTLENS_OPENAI_BASE_URL", None)
_default_models = Path.home() / ".cache" / "textlens" / "models"
_models_dir = os.environ.get("TEXTLENS_TEST_MODELS_DIR") or (str(_default_models) if (_default_models / "ppocrv6-small").is_dir() else str(_HOME / "models"))
os.environ["TEXTLENS_MODELS_DIR"] = _models_dir

sys.path.insert(0, str(Path(__file__).parent))

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "models: needs real model weights (PP-OCRv6); skipped if unavailable")
    config.addinivalue_line("markers", "slow: slower integration tests")


def _ppocr_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    from textlens.models import artifacts
    from textlens.models.specs import get_spec

    spec = get_spec("ppocrv6-small")
    if artifacts.is_installed(spec):
        return True
    if os.environ.get("TEXTLENS_TEST_DOWNLOAD") == "1":
        try:
            artifacts.install(spec)
            return True
        except Exception:
            return False
    return False


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    if not any("models" in item.keywords for item in items):
        return
    if _ppocr_available():
        return
    skip = pytest.mark.skip(reason="PP-OCRv6 weights not installed (set TEXTLENS_TEST_MODELS_DIR or TEXTLENS_TEST_DOWNLOAD=1)")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_state():
    """Reset process-wide caches between tests."""
    from textlens.config import reload_settings
    from textlens.runtime import cache as cache_mod

    reload_settings()
    cache_mod._shared.clear()
    yield
    cache_mod._shared.clear()


# ── fake engines ─────────────────────────────────────────────────────────────

FAKE_CALLS: Dict[str, List[Any]] = {}


def _reset_calls() -> None:
    FAKE_CALLS.clear()


def make_fake_backend_class(text: str = "FAKE TEXT LINE", confidence: float = 0.95, markdown: bool = False, fail: bool = False):
    from textlens.backends.base import OCRBackend, PageOCR
    from textlens.core.result import BBox
    from textlens.documents.layout import Item

    class _Fake(OCRBackend):
        thread_safe = True
        backend_name = "fake"

        def _load(self) -> None:
            self.device = "cpu"

        def _recognize(self, images, options):  # type: ignore[no-untyped-def]
            FAKE_CALLS.setdefault(self.spec.id, []).append(len(images))
            if fail:
                from textlens.errors import InferenceError

                raise InferenceError(f"{self.spec.id} failed on purpose")
            out = []
            for img in images:
                w, h = img.size
                if markdown:
                    out.append(PageOCR(width=w, height=h, markdown=f"# Title\n\n{text}\n\n| a | b |\n|---|---|\n| 1 | 2 |", confidence=confidence))
                else:
                    items = [
                        Item(text=line, bbox=BBox(10, 10 + 30 * i, w * 0.8, 34 + 30 * i), confidence=confidence)
                        for i, line in enumerate(text.split("\n"))
                    ]
                    out.append(PageOCR(width=w, height=h, items=items, confidence=confidence))
            return out

    return _Fake


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the catalog with deterministic fake engines.

    Returns a helper ``register(id, **kwargs)`` that adds a fake model.
    ``fake-fast`` (onnx-like, high confidence) is registered by default.
    """
    import textlens.models.specs as specs_mod
    from textlens.models.specs import ModelSpec
    from textlens.runtime.system import set_system_info

    _reset_calls()
    saved_specs = dict(specs_mod._SPECS)
    saved_aliases = dict(specs_mod._ALIASES)
    specs_mod._SPECS.clear()
    specs_mod._ALIASES.clear()
    classes: Dict[str, Any] = {}
    module = sys.modules[__name__]

    def register(model_id: str, *, text: str = "FAKE TEXT LINE", confidence: float = 0.95, generative: bool = False,
                 fail: bool = False, quality: int = 3, speed: int = 4, default: bool = False, **spec_kwargs: Any) -> ModelSpec:
        cls = make_fake_backend_class(text=text, confidence=confidence, markdown=generative, fail=fail)
        attr = "Fake_" + model_id.replace("-", "_").replace(".", "_")
        setattr(module, attr, cls)
        classes[model_id] = cls
        tasks = frozenset({"text", "markdown", "table", "formula"}) if generative else frozenset({"text"})
        fields: Dict[str, Any] = dict(
            id=model_id,
            display_name=model_id,
            family="fake",
            provider="tests",
            description="fake engine for tests",
            backend="transformers" if generative else "onnx",
            adapter=f"conftest:{attr}",
            tasks=tasks,
            markdown=generative,
            confidence=not generative,
            cpu=True,
            cpu_practical=True,
            edge=not generative,
            quality_tier=quality,
            speed_tier=speed,
            is_default=default,
            license="apache-2.0",
            commercial_use="yes",
        )
        fields.update(spec_kwargs)  # explicit overrides win
        spec = ModelSpec(**fields)
        specs_mod.register_spec(spec, replace=True)
        return spec

    # Fake models are "installed" and their runtimes "present".
    monkeypatch.setattr("textlens.models.artifacts.is_installed", lambda spec, root=None: True)
    monkeypatch.setattr("textlens.backends.loader.missing_requirements", lambda spec: [])
    from textlens.backends.loader import get_pool

    get_pool().unload()
    set_system_info(fake_system())
    register("fake-fast", default=True)
    yield register
    get_pool().unload()
    specs_mod._SPECS.clear()
    specs_mod._SPECS.update(saved_specs)
    specs_mod._ALIASES.clear()
    specs_mod._ALIASES.update(saved_aliases)
    set_system_info(None)


def fake_system(gpu_vram: float = 0.0, torch_cuda: bool = False, device_class: str = "desktop", ram: float = 16.0):
    from textlens.models.hardware import GPUInfo, HardwareProfile
    from textlens.runtime.system import PlatformInfo, RuntimeInfo, SystemInfo

    gpus = [GPUInfo(0, "Test GPU", gpu_vram, gpu_vram, True)] if gpu_vram else []
    hw = HardwareProfile(
        os_name="TestOS", python_version="3.12", torch_version="2.5" if torch_cuda else "Not Installed",
        cuda_available=bool(gpus), cuda_version=None, system_cuda_version="12.4" if gpus else None, gpus=gpus,
        primary_gpu_name="Test GPU" if gpus else None, primary_vram_gb=gpu_vram, cpu_name="Test CPU",
        cpu_physical_cores=4, cpu_logical_cores=8, ram_total_gb=ram, device_type="cuda" if gpus else "cpu",
    )
    plat = PlatformInfo(system="Linux", machine="x86_64" if device_class == "desktop" else "aarch64", python="3.12", device_class=device_class)
    rt = RuntimeInfo(onnxruntime="1.20", ort_providers=("CPUExecutionProvider",), torch="2.5+cu124" if torch_cuda else None,
                     torch_cuda_build=True if torch_cuda else None, transformers="5.0" if torch_cuda else None)
    return SystemInfo(hardware=hw, platform=plat, runtimes=rt)


@pytest.fixture
def fixture_pdf() -> bytes:
    import pdf_factory as F

    return F.build_pdf(
        [
            F.heading_and_body_page("Annual Report 2026", "Introduction", F.LOREM),
            F.scanned_page(["Scanned invoice 4471", "Total due: 5,240.00 EUR"]),
            F.scanned_with_ocr_layer(F.LOREM),
            F.garbled_text_page(F.LOREM),
            F.blank_page(),
            F.ruled_table_page([["Item", "Qty", "Price"], ["Apple", "3", "1.20"], ["Pear", "5", "0.80"]]),
        ],
        info={"Title": "Fixture Report", "Author": "TextLens Tests"},
    )
