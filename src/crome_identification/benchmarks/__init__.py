"""Auditable benchmark generators for the CS evidence package."""

from .aligned_geometry import AlignedGeometry, PerturbationSpec, generate_aligned_geometry
from .online_retail import RetailEvent, preprocess_rows

__all__ = [
    "AlignedGeometry", "PerturbationSpec", "RetailEvent",
    "generate_aligned_geometry", "preprocess_rows",
]
