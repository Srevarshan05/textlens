"""
textlens.anpr
─────────────
Number-plate recognition, kept separate from the core OCR system.

    from textlens import ANPR
    result = ANPR(profile="edge", region="in")("vehicle.jpg")
    for plate in result.plates:
        print(plate.plate, plate.confidence, plate.valid)
"""

from textlens.anpr.formats import REGIONS, PlateFormat, Region, load_region
from textlens.anpr.pipeline import ANPR
from textlens.anpr.types import ANPRResult, PlateRead
from textlens.anpr.validation import PlateValidator, normalize

__all__ = ["ANPR", "ANPRResult", "PlateRead", "PlateFormat", "Region", "REGIONS", "load_region", "PlateValidator", "normalize"]
