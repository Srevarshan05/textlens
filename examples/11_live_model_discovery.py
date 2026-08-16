"""Find live Hugging Face OCR/VLM candidates that match this computer.

Install the small discovery dependency first:
    python -m pip install "textlens-ocr[catalog]"

This does not download models or make a repository an officially supported
TextLens backend. It is a research aid before choosing or integrating a model.
"""

from textlens.models import discover_models, inspect_hardware
from textlens.models.discovery import print_discovered_models


def main() -> None:
    profile = inspect_hardware()
    candidates = discover_models(
        search="ocr",
        use_case="invoices",
        limit=10,
        compatible_only=True,
        profile=profile,
    )
    if candidates:
        print_discovered_models(candidates, profile)
    else:
        print("No compatible candidates reported by the live Hub search.")


if __name__ == "__main__":
    main()
