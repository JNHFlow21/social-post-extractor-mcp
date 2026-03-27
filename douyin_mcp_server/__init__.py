"""Backward-compatible shim for the renamed package."""

from social_post_extractor_mcp import __author__, __email__, __version__, main

__all__ = ["main", "__version__", "__author__", "__email__"]
