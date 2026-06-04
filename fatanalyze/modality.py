"""Modality enum for CT vs MR analysis mode."""
from __future__ import annotations

from enum import Enum


class Modality(Enum):
    CT = "ct"
    MR = "mr"
