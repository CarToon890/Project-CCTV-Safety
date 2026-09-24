"""Versioned detector schema shared by data preparation and inference."""

DETECTOR_SCHEMA_VERSION = 2
CLASS_NAMES_V2 = ("person", "helmet", "vest", "fall", "fire", "smoke")
CLASS_TO_ID_V2 = {name: index for index, name in enumerate(CLASS_NAMES_V2)}

# Kept only so loaders can identify and reject legacy checkpoints explicitly.
DETECTOR_SCHEMA_VERSION_LEGACY = 1
CLASS_NAMES_V1 = (*CLASS_NAMES_V2, "fight")
CLASS_TO_ID_V1 = {name: index for index, name in enumerate(CLASS_NAMES_V1)}

CLASS_NAMES = CLASS_NAMES_V2
CLASS_TO_ID = CLASS_TO_ID_V2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
