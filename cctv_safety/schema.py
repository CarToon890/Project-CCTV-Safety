"""Canonical detector schema shared by data preparation and inference."""

CLASS_NAMES = ("person", "helmet", "vest", "fall", "fire", "smoke", "fight")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

