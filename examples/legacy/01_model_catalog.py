"""List supported models and inspect one model's hardware recommendation."""

from textlens import ModelManager


def main() -> None:
    ModelManager.models()
    metadata = ModelManager.info("glm-ocr")
    print(f"\nDefault model: {metadata.id}")
    print(f"Cache installed: {ModelManager.is_installed(metadata.id)}")


if __name__ == "__main__":
    main()
